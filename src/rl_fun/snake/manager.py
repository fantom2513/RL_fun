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
        self.human = SnakeEnv(data.get("size", 16)) if self.mode != "agent" else None
        self.agent = SnakeEnv(data.get("size", 16)) if self.mode != "manual" else None
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
        if self.agent and not self.agent.done:
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

    def models(self):
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
                        }
                    )
                except (OSError, ValueError, KeyError, TypeError):
                    continue
            return sorted(rows, key=lambda row: row["created"], reverse=True)[:100]

    def state(self):
        with self.condition:
            return {"run": copy.deepcopy(self.run), "revision": self.revision}

    def train(self, config_data, resume=None):
        config = SnakeConfig.from_dict(config_data)
        resume_path = None
        if resume:
            from rl_fun.snake.checkpoint import load_checkpoint

            resume_path = self._path(resume) / "model.pt"
            restored = load_checkpoint(resume_path)
            previous = restored["config"]
            keys = ["algorithm", "size"]
            if config.algorithm == "evolution":
                keys.append("population")
            if any(previous[key] != config.to_dict()[key] for key in keys):
                raise ValueError("для продолжения нужны прежний алгоритм, поле и популяция")
            budget = "episodes" if config.algorithm == "dqn" else "generations"
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
            self._persist()
            parent, child = self.context.Pipe()
            process = self.context.Process(
                target=run_worker,
                args=(
                    config.to_dict(),
                    child,
                    str(path),
                    str(resume_path) if resume_path else None,
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
                with self.condition:
                    if not self.run or self.run["id"] != run_id:
                        break
                    if message["kind"] == "metric":
                        self.run["history"].append(message["metric"])
                        self.run["history"] = self.run["history"][-500:]
                    elif message["kind"] == "saved":
                        self.run["saved"] = True
                        self.run["saved_iteration"] = message["iteration"]
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
                if data.get("model"):
                    from rl_fun.snake.checkpoint import load_checkpoint

                    state = load_checkpoint(self._path(data["model"]) / "model.pt")
                    if state["config"]["size"] != data.get("size", 16):
                        raise ValueError("выберите размер поля, с которым обучалась модель")
                    policy.load_state_dict(state["model"])
                    trained = (
                        state.get("updates", 0) > 0
                        if state["algorithm"] == "dqn"
                        else state["completed"] > 0
                    )
            session = SnakeSession(data, policy, trained)
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
            return self.sessions[game_id].step(data)

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
