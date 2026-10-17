"""Bounded interactive sessions and ownership of one Snake training process."""

from __future__ import annotations

import copy
import json
import multiprocessing
import os
import re
import threading
import time
import uuid
from pathlib import Path

from rl_fun.snake.config import SnakeConfig
from rl_fun.snake.env import SnakeEnv, relative_action
from rl_fun.snake.worker import run_worker

TERMINAL = {"finished", "stopped", "error"}


class SnakeSession:
    def __init__(self, data, policy=None, trained=False):
        self.id = uuid.uuid4().hex
        self.mode = data.get("mode", "manual")
        if self.mode not in ("manual", "agent", "split"):
            raise ValueError("режим: manual, agent или split")
        self.seed = data.get("seed", 42)
        if type(self.seed) is not int or not 0 <= self.seed <= 2**31 - 1:
            raise ValueError("seed должен быть целым числом от 0 до 2147483647")
        self.policy = policy
        self.trained = trained
        self.model_id = data.get("model") or None
        self.follow_training = data.get("follow_training", False)
        if type(self.follow_training) is not bool:
            raise ValueError("follow_training должен быть логическим значением")
        self.model_iteration = 0
        self.model_revision = None
        self.agent_episode = 1
        self.human = SnakeEnv(data.get("size", 16)) if self.mode != "agent" else None
        self.agent = (
            SnakeEnv(
                data.get("size", 16), observation_mode=getattr(policy, "observation_mode", "local")
            )
            if self.mode != "manual"
            else None
        )
        for env in (self.human, self.agent):
            if env is not None:
                env.reset(seed=self.seed)
        self.last_used = time.monotonic()

    def snapshot(self):
        return {
            "id": self.id,
            "mode": self.mode,
            "seed": self.seed,
            "human": self.human.snapshot() if self.human else None,
            "agent": self.agent.snapshot() if self.agent else None,
            "trained": self.trained,
            "model": self.model_id,
            "model_iteration": self.model_iteration,
            "agent_episode": self.agent_episode,
            "follow_training": self.follow_training,
        }

    def step(self, data):
        direction = data.get("direction")
        if direction is not None:
            relative_action(0, direction)  # validate even if the human side has ended
        if self.human and not self.human.done:
            action = (
                relative_action(self.human.direction, direction) if direction is not None else 0
            )
            self.human.step(action)
        if self.agent and self.agent.reason != "win":
            if self.agent.done:
                self.agent_episode += 1
                self.agent.reset(seed=(self.seed + self.agent_episode - 1) % (2**31))
            self.agent.step(self.policy.act(self.agent.observation()))
        self.last_used = time.monotonic()
        return self.snapshot()


class SnakeManager:
    def __init__(self, runs_dir):
        self.root = Path(runs_dir).resolve() / "snake"
        self.root.mkdir(parents=True, exist_ok=True)
        self.condition = threading.Condition(threading.RLock())
        self.run = None
        self.process = self.connection = self.reader = None
        self.sessions = {}
        self.closed = False
        self.revision = 0
        self.frame_revision = 0
        self.latest_frame = None
        self.visualization = {"enabled": False, "fps": 8}
        self.context = multiprocessing.get_context("spawn")

    def _path(self, model_id):
        if not isinstance(model_id, str) or not re.fullmatch(r"[a-f0-9]{32}", model_id):
            raise ValueError("некорректный идентификатор модели")
        path = (self.root / model_id).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("модель вне каталога запусков")
        return path

    def _notify(self):
        self.revision += 1
        self.condition.notify_all()

    def _persist(self):
        path = self._path(self.run["id"]) / "record.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.run, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)

    def models(self, limit=100):
        with self.condition:
            rows = []
            for directory in self.root.iterdir():
                if not re.fullmatch(r"[a-f0-9]{32}", directory.name):
                    continue
                try:
                    path = self._path(directory.name)
                    if not (path / "model.pt").is_file():
                        continue
                    record = json.loads((path / "record.json").read_text(encoding="utf-8"))
                    verification = record.get("verification")
                    if (
                        verification
                        and verification.get("model_mtime_ns")
                        != (path / "model.pt").stat().st_mtime_ns
                    ):
                        verification = None
                    rows.append(
                        {
                            "id": directory.name,
                            "algorithm": record["config"]["algorithm"],
                            "config": record["config"],
                            "size": record["config"]["size"],
                            "created": record["created"],
                            "status": record["status"]
                            if record["status"] in TERMINAL
                            else "interrupted",
                            "iteration": record.get("saved_iteration", 0),
                            "history": record.get("history", [])[-500:],
                            "verification": verification,
                        }
                    )
                except (OSError, ValueError, KeyError, TypeError):
                    continue
            rows.sort(key=lambda row: row["created"], reverse=True)
            from rl_fun.snake.gpu.pretrained import pretrained_rows

            return (rows if limit is None else rows[:limit]) + pretrained_rows()

    def presets(self):
        from rl_fun.snake.presets import completion_preset, verification_status

        presets = [completion_preset()]
        models = self.models(limit=None)
        measured = []
        for model in models:
            if model.get("builtin"):
                continue
            try:
                directory = self._path(model["id"])
                report = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
                if (
                    report["phase"] == "tuning"
                    and model["size"] >= 16
                    and model["config"]["observation_mode"] in ("board", "egocentric")
                    and report["config"] == model["config"]
                    and report["model_mtime_ns"] == (directory / "model.pt").stat().st_mtime_ns
                    and report["episodes"] == 20
                    and 0 <= report["wins"] <= 20
                    and 0 <= report["occupancy"] <= 1
                ):
                    measured.append((report["wins"], report["occupancy"], model, report))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        if measured:
            _, _, model, report = max(measured, key=lambda row: row[:2])
            presets.append(
                {
                    "id": "measured_candidate",
                    "model": model["id"],
                    "label": f"Лучший в поиске · {model['algorithm']} · экспериментальный",
                    "status": "experimental",
                    "config": model["config"],
                    "visualization": {"enabled": False, "fps": 60},
                    "description": f"Настройка: {report['wins']}/20 побед, "
                    f"среднее заполнение {report['occupancy']:.1%}. "
                    "Выбрана сохранённая модель. Независимая проверка ещё не пройдена.",
                }
            )
            labels = {
                "dqn": "DQN",
                "double_dqn": "Double DQN",
                "reinforce": "REINFORCE / Policy Gradient",
                "a2c": "A2C",
                "ppo": "PPO",
                "td3": "TD3 · адаптер ходов",
                "evolution": "Эволюция",
            }
            for algorithm, label in labels.items():
                candidates = [row for row in measured if row[2]["algorithm"] == algorithm]
                if not candidates:
                    continue
                _, _, model, report = max(candidates, key=lambda row: row[:2])
                presets.append(
                    {
                        "id": "best_" + algorithm,
                        "model": model["id"],
                        "label": f"{label} · {report['occupancy']:.1%} поля · "
                        f"{report['wins']}/20 побед",
                        "status": "experimental",
                        "config": SnakeConfig.from_dict(model["config"]).to_dict(),
                        "visualization": {"enabled": False, "fps": 60},
                        "result": {key: report[key] for key in ("wins", "episodes", "occupancy")},
                        "description": f"Лучший оценённый {label} "
                        f"на поле {model['size']}×{model['size']}: "
                        f"{report['wins']}/20 побед, среднее заполнение {report['occupancy']:.1%}. "
                        "Загружены обученные веса и их параметры. "
                        "Результат на играх для настройки.",
                    }
                )
        for model in models:
            if model.get("builtin"):
                evidence = model["verification"]
                presets.append(
                    {
                        "id": f"pretrained_{model['size']}",
                        "model": model["id"],
                        "label": f"Полное поле {model['size']}×{model['size']} · "
                        "предобученная нейросеть",
                        "status": evidence["status"],
                        "config": model["config"],
                        "visualization": {"enabled": False, "fps": 60},
                        "description": f"Предобучена имитацией гамильтонова цикла: "
                        f"{evidence['wins']}/{evidence['episodes']} побед на независимых играх. "
                        "Играет жадно, без маски ходов и без учителя. Только показ модели: "
                        "продолжить её обучение нельзя.",
                    }
                )
                continue
            evidence = model.get("verification")
            if not evidence or evidence.get("status") != "verified":
                continue
            if (
                verification_status(model["size"], evidence["episodes"], evidence["wins"])
                != "verified"
            ):
                continue
            presets.append(
                {
                    "id": "verified_" + model["id"],
                    "model": model["id"],
                    "label": f"Полное поле {model['size']}×{model['size']} · проверенный",
                    "status": "verified",
                    "config": model["config"],
                    "visualization": {"enabled": False, "fps": 60},
                    "description": f"Проверено: {evidence['wins']}/{evidence['episodes']} побед. "
                    "Выбрана обученная модель; можно играть или продолжить обучение.",
                }
            )
        return presets

    def state(self):
        with self.condition:
            return {
                "run": copy.deepcopy(self.run),
                "revision": self.revision,
                "frame": copy.deepcopy(self.latest_frame),
                "frame_revision": self.frame_revision,
                "visualization": dict(self.visualization),
            }

    @staticmethod
    def _visualization(data):
        if (
            not isinstance(data, dict)
            or set(data) != {"enabled", "fps"}
            or type(data["enabled"]) is not bool
            or type(data["fps"]) is not int
            or not 1 <= data["fps"] <= 240
        ):
            raise ValueError("для анимации нужны enabled (да/нет) и fps (1–240)")
        return dict(data)

    def configure_visualization(self, data):
        settings = self._visualization(data)
        with self.condition:
            if self.run and self.run["status"] not in TERMINAL:
                try:
                    self.connection.send({"command": "visualization", "settings": settings})
                except (BrokenPipeError, OSError):
                    raise RuntimeError("процесс обучения недоступен") from None
            self.visualization = settings
            self._notify()

    def train(self, config_data, resume=None, visualization=None):
        config = SnakeConfig.from_dict(config_data)
        settings = self._visualization(
            {"enabled": False, "fps": 8} if visualization is None else visualization
        )
        resume_path = None
        if resume:
            from rl_fun.snake.checkpoint import load_checkpoint

            resume_path = self._path(resume) / "model.pt"
            restored = load_checkpoint(resume_path)
            previous = SnakeConfig.from_dict(restored["config"]).to_dict()
            keys = ["algorithm", "size", "observation_mode"]
            if config.algorithm == "evolution":
                keys.append("population")
            if any(previous[key] != config.to_dict()[key] for key in keys):
                raise ValueError(
                    "для продолжения нужны прежний алгоритм, поле, наблюдение и популяция"
                )
            budget = "generations" if config.algorithm == "evolution" else "episodes"
            SnakeConfig.from_dict(
                {**config.to_dict(), budget: getattr(config, budget) + restored["completed"]}
            )
        with self.condition:
            if self.closed:
                raise RuntimeError("сервер останавливается")
            if self.run and self.run["status"] not in TERMINAL:
                raise RuntimeError("сначала остановите текущее обучение змейки")
            if self.process:
                self.process.join(1)
                if self.process.is_alive():
                    raise RuntimeError("предыдущий процесс ещё завершает работу")
            run_id = uuid.uuid4().hex
            path = self._path(run_id)
            path.mkdir()
            self.run = {
                "id": run_id,
                "config": config.to_dict(),
                "created": time.time(),
                "status": "running",
                "history": [],
                "saved": False,
                "saved_iteration": 0,
                "error": None,
            }
            self.visualization = settings
            self.latest_frame = None
            self.frame_revision += 1
            self._persist()
            parent, child = self.context.Pipe()
            process = self.context.Process(
                target=run_worker,
                args=(
                    config.to_dict(),
                    child,
                    str(path),
                    str(resume_path) if resume_path else None,
                    settings,
                ),
                daemon=True,
                name=f"snake-{run_id[:8]}",
            )
            try:
                process.start()
            except Exception:
                parent.close()
                child.close()
                self.run["status"] = "error"
                self._persist()
                raise
            child.close()
            self.process, self.connection = process, parent
            self.reader = threading.Thread(
                target=self._read,
                args=(run_id, parent, process),
                daemon=True,
                name=f"snake-reader-{run_id[:8]}",
            )
            self.reader.start()
            self._notify()
            return run_id

    def _read(self, run_id, connection, process):
        try:
            while True:
                if not connection.poll(0.2):
                    if not process.is_alive():
                        break
                    continue
                message = connection.recv()
                if message["kind"] == "status" and message["status"] in TERMINAL:
                    # A reusable run must have released its child process as well as saved weights.
                    process.join()
                with self.condition:
                    if not self.run or self.run["id"] != run_id:
                        break
                    if message["kind"] == "frame":
                        self.frame_revision += 1
                        self.latest_frame = {
                            "run_id": run_id,
                            "seq": self.frame_revision,
                            **message["frame"],
                        }
                        self.condition.notify_all()
                        continue
                    elif message["kind"] == "metric":
                        self.run["history"].append(message["metric"])
                        self.run["history"] = self.run["history"][-500:]
                    elif message["kind"] == "saved":
                        self.run["saved"] = True
                        self.run["saved_iteration"] = message["iteration"]
                        self.run["saved_revision"] = self.run.get("saved_revision", 0) + 1
                    elif message["kind"] == "status":
                        self.run["status"] = message["status"]
                        self.run["error"] = message.get("error")
                    self._persist()
                    self._notify()
        except (EOFError, OSError):
            pass
        finally:
            connection.close()
            with self.condition:
                if self.run and self.run["id"] == run_id and self.run["status"] not in TERMINAL:
                    self.run["status"] = "error"
                    self.run["error"] = "процесс обучения завершился неожиданно"
                    self._persist()
                    self._notify()

    def control(self, command):
        if command not in ("pause", "resume", "stop", "save"):
            raise ValueError("команда: pause, resume, stop или save")
        with self.condition:
            if not self.run or self.run["status"] in TERMINAL:
                if command == "save" and self.run and self.run["saved"]:
                    return
                raise ValueError("нет активного обучения")
            if command == "pause" and self.run["status"] != "running":
                raise ValueError("обучение уже на паузе")
            if command == "resume" and self.run["status"] != "paused":
                raise ValueError("обучение не на паузе")
            try:
                self.connection.send(command)
            except (BrokenPipeError, OSError):
                raise RuntimeError("процесс обучения недоступен") from None

    def create_game(self, data):
        if not isinstance(data, dict):
            raise ValueError("параметры игры должны быть объектом")
        mode = data.get("mode", "manual")
        if mode not in ("manual", "agent", "split"):
            raise ValueError("неизвестный режим игры")
        policy = None
        trained = False
        state = None
        with self.condition:
            if self.closed:
                raise RuntimeError("сервер останавливается")
            if mode != "manual":
                import torch

                from rl_fun.snake.policy import Policy

                torch.set_num_threads(1)
                with torch.random.fork_rng():
                    torch.manual_seed(42)
                    policy = Policy()
                from rl_fun.snake.gpu.pretrained import PRETRAINED, load_pretrained

                if isinstance(data.get("model"), str) and data["model"] in PRETRAINED:
                    policy = load_pretrained(data["model"])
                    if policy.size != data.get("size", 16):
                        raise ValueError("выберите размер поля, с которым обучалась модель")
                    trained = True
                elif data.get("model"):
                    from rl_fun.snake.checkpoint import load_checkpoint, playback_weights

                    state = load_checkpoint(self._path(data["model"]) / "model.pt")
                    if state["config"]["size"] != data.get("size", 16):
                        raise ValueError("выберите размер поля, с которым обучалась модель")
                    policy = Policy(
                        state["config"]["size"], state["config"].get("observation_mode", "local")
                    )
                    policy.load_state_dict(playback_weights(state))
                    policy.action_adapter = state.get("action_adapter")
                    trained = (
                        state.get("actor_updates" if state["algorithm"] == "td3" else "updates", 0)
                        > 0
                        if state["algorithm"] != "evolution"
                        else state["completed"] > 0
                    )
            session = SnakeSession(data, policy, trained)
            if state is not None and mode != "manual":
                session.model_iteration = state["completed"]
            self._refresh_policy(session)
            expired = [
                key
                for key, value in self.sessions.items()
                if time.monotonic() - value.last_used > 600
            ]
            for key in expired:
                del self.sessions[key]
            while len(self.sessions) >= 32:
                del self.sessions[next(iter(self.sessions))]
            self.sessions[session.id] = session
            return session.snapshot()

    def step_game(self, game_id, data):
        with self.condition:
            session = self.sessions[game_id]
            if "follow_training" in data:
                if type(data["follow_training"]) is not bool:
                    raise ValueError("follow_training должен быть логическим значением")
                session.follow_training = data["follow_training"]
            self._refresh_policy(session)
            return session.step(data)

    def _refresh_policy(self, session):
        run = self.run
        if not (session.agent and session.follow_training and run and run.get("saved")):
            return
        if run["config"]["size"] != session.agent.size:
            return
        revision = (run["id"], run.get("saved_revision", run["saved_iteration"]))
        if session.model_revision == revision:
            return
        from rl_fun.snake.checkpoint import load_checkpoint, playback_weights
        from rl_fun.snake.policy import Policy

        state = load_checkpoint(self._path(run["id"]) / "model.pt")
        observation_mode = state["config"].get("observation_mode", "local")
        session.policy = Policy(session.agent.size, observation_mode)
        session.policy.load_state_dict(playback_weights(state))
        session.policy.action_adapter = state.get("action_adapter")
        if session.agent.observation_mode != observation_mode:
            session.agent = SnakeEnv(session.agent.size, observation_mode=observation_mode)
            session.agent.reset(seed=session.seed)
        session.model_id = run["id"]
        session.model_iteration = state["completed"]
        session.model_revision = revision
        session.trained = (
            state.get("actor_updates" if state["algorithm"] == "td3" else "updates", 0) > 0
            if state["algorithm"] != "evolution"
            else state["completed"] > 0
        )

    def delete_game(self, game_id):
        with self.condition:
            del self.sessions[game_id]

    def shutdown(self):
        with self.condition:
            if self.closed:
                return
            self.closed = True
            if self.process and self.process.is_alive():
                try:
                    self.connection.send("stop")
                except (OSError, EOFError):
                    pass
            self._notify()
        if self.process:
            self.process.join(4)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
        if self.reader:
            self.reader.join(2)
        self.sessions.clear()
