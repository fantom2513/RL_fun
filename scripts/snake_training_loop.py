"""Continuously train, screen, and preserve Snake completion candidates."""

import argparse
import json
import os
import signal
import time
from contextlib import contextmanager
from pathlib import Path

from rl_fun.snake.benchmark import run_benchmark, write_json
from rl_fun.snake.checkpoint import load_checkpoint
from rl_fun.snake.config import SnakeConfig
from rl_fun.snake.manager import SnakeManager

ALGORITHM_SCHEDULE = ("a2c", "ppo", "double_dqn", "reinforce", "ppo", "td3", "dqn", "ppo")


@contextmanager
def exclusive_loop(path):
    """The OS releases this lock even when the loop process crashes."""
    with path.open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def trial_config(base, cycle):
    """Use comparable parameter variants; keep DQN replay reward semantics."""
    data = {**base, "episodes": 100000, "evaluation_interval": 200, "evaluation_episodes": 20}
    variant = cycle % 6
    if data["algorithm"] in ("ppo", "a2c", "reinforce"):
        data.update(
            learning_rate=(0.00003, 0.0001, 0.00001, 0.00003, 0.0001, 0.00003)[variant],
            gamma=0.999,
            gae_lambda=(0.99, 0.99, 0.995, 0.99, 0.995, 0.99)[variant],
            entropy_coefficient=(0.01, 0.02, 0.02, 0.01, 0.005, 0.03)[variant],
            distance_reward=(0, 0, 0.02, 0.02, 0, 0)[variant],
            curriculum_steps=(2000000, 0, 2000000, 0, 0, 2000000)[variant],
            step_penalty=0.001,
            idle_steps=1024,
            idle_penalty=0.001,
            rollout_steps=max(base["rollout_steps"], (2048, 1024, 4096, 2048, 512, 2048)[variant]),
            ppo_clip=(0.1, 0.2, 0.1, 0.15, 0.2, 0.1)[variant],
        )
    elif data["algorithm"] == "td3":
        data.update(
            learning_rate=(0.0001, 0.00003, 0.0003, 0.0001, 0.00003, 0.0001)[variant],
            gamma=(0.99, 0.999, 0.995, 0.999, 0.995, 0.999)[variant],
            action_noise=(0.1, 0.2, 0.05, 0.15, 0.1, 0.3)[variant],
            target_noise=(0.1, 0.2, 0.1, 0.15, 0.05, 0.2)[variant],
            policy_delay=(2, 2, 3, 2, 3, 4)[variant],
            tau=(0.005, 0.001, 0.005, 0.01, 0.001, 0.005)[variant],
            curriculum_steps=(0, 2000000, 0, 2000000, 0, 0)[variant],
        )
    else:
        data.update(
            learning_rate=(0.0001, 0.00003, 0.0001, 0.00003, 0.0001, 0.00003)[variant],
            gamma=(0.999, 0.999, 0.995, 0.999, 0.999, 0.995)[variant],
            epsilon_end=(0.02, 0.05, 0.02, 0.1, 0.05, 0.02)[variant],
            target_interval=(500, 1000, 500, 1000, 200, 500)[variant],
            curriculum_steps=(2000000, 0, 0, 2000000, 0, 0)[variant],
        )
    return SnakeConfig.from_dict(data)


def run_loop(runs_dir, seconds=1200, steps=1000000, max_cycles=0, target_occupancy=0.5):
    root = Path(runs_dir).resolve() / "snake"
    root.mkdir(parents=True, exist_ok=True)
    state_path, stop_path = root / "training-loop.json", root / "training-loop.stop"
    with exclusive_loop(root / "training-loop.lock"):
        if stop_path.exists():
            raise RuntimeError("Remove training-loop.stop before explicitly restarting the loop")
        previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        if previous.get("validation"):
            raise RuntimeError(
                "Independent validation already consumed; review it before a new search"
            )
        latest_models = dict(previous.get("latest_models", {}))
        for record in previous.get("history", []):
            report_path = Path(record["directory"]) / "evaluation.json"
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                latest_models[report["config"]["algorithm"]] = report_path.parent.name
        state = {
            "pid": os.getpid(),
            "status": "running",
            "started": time.time(),
            "size": 16,
            "cycle": previous.get("cycle", 0),
            "history": previous.get("history", []),
            "latest_models": latest_models,
            "algorithm_trials": previous.get("algorithm_trials", {}),
            "schedule_cursor": previous.get("schedule_cursor", 0),
            "algorithm_schedule": list(ALGORITHM_SCHEDULE),
            "seconds_per_cycle": seconds,
            "steps_per_cycle": steps,
            "stop_file": str(stop_path),
            "target": "Mean board occupancy on 100 frozen independent validation games",
            "target_occupancy": target_occupancy,
        }
        manager = SnakeManager(runs_dir)

        def stop(*_):
            stop_path.touch()

        handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        write_json(state_path, state)
        try:
            for _ in range(max_cycles) if max_cycles else iter(int, 1):
                if stop_path.exists():
                    state["status"] = "stopped"
                    break
                seeds = {p["id"]: p for p in manager.presets()}
                algorithm = ALGORITHM_SCHEDULE[state["schedule_cursor"] % len(ALGORITHM_SCHEDULE)]
                seed = seeds.get("best_" + algorithm)
                source = state["latest_models"].get(algorithm) or (seed["model"] if seed else None)
                resume = root / source / "model.pt" if source else None
                base = (
                    load_checkpoint(resume)["config"]
                    if resume
                    else SnakeConfig(
                        algorithm=algorithm, observation_mode="board", size=16, max_steps=100000
                    ).to_dict()
                )
                if base["size"] != 16 or base["observation_mode"] not in ("board", "egocentric"):
                    resume, source = None, None
                    base = SnakeConfig(
                        algorithm=algorithm, observation_mode="board", size=16, max_steps=100000
                    ).to_dict()
                trial = state["algorithm_trials"].get(algorithm, 0)
                config = trial_config(base, trial)
                if resume and load_checkpoint(resume)["completed"] >= config.episodes:
                    resume = None
                state.update(
                    algorithm=algorithm,
                    size=config.size,
                    config=config.to_dict(),
                    source_model=source,
                    parameter_variant=trial % 6,
                )
                write_json(state_path, state)
                print(
                    f"Cycle {state['cycle'] + 1}: {algorithm}, "
                    f"variant={trial % 6}, source={source}",
                    flush=True,
                )
                directory = run_benchmark(
                    config,
                    runs_dir,
                    seconds,
                    resume,
                    train_steps=steps,
                    stop_requested=stop_path.exists,
                )
                report = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
                state["cycle"] += 1
                state["schedule_cursor"] += 1
                state["algorithm_trials"][algorithm] = trial + 1
                state["latest_models"][algorithm] = directory.name
                state["history"].append(
                    {
                        "cycle": state["cycle"],
                        "directory": str(directory),
                        "algorithm": algorithm,
                        "parameter_variant": trial % 6,
                        **{k: report[k] for k in ("wins", "episodes", "occupancy", "steps_added")},
                    }
                )
                write_json(state_path, state)
                if stop_path.exists():
                    state["status"] = "stopped"
                    break
                if report["occupancy"] >= target_occupancy:
                    state["status"] = "validating"
                    write_json(state_path, state)
                    frozen = directory / "model.pt"
                    frozen_config = SnakeConfig.from_dict(load_checkpoint(frozen)["config"])
                    validation = run_benchmark(
                        frozen_config, runs_dir, resume=frozen, validate=True
                    )
                    evidence = json.loads(
                        (validation / "validation.json").read_text(encoding="utf-8")
                    )
                    state["validation"] = str(validation)
                    state["status"] = (
                        "verified_occupancy_target"
                        if evidence["occupancy"] >= target_occupancy
                        else "validation_needs_review"
                    )
                    # Never feed repeated held-out results into the parameter search.
                    break
            else:
                state["status"] = "budget_finished"
        except Exception as error:
            state.update(status="error", error=str(error))
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            state["updated"] = time.time()
            write_json(state_path, state)
            manager.shutdown()
            print(f"Loop status: {state['status']}; state={state_path}", flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs/lab")
    parser.add_argument("--seconds", type=float, default=1200)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--target-occupancy", type=float, default=0.5)
    parser.add_argument(
        "--max-cycles", type=int, default=0, help="0 runs until stopped or validated"
    )
    args = parser.parse_args()
    if not 0 < args.seconds < float("inf") or args.steps < 1 or args.max_cycles < 0:
        parser.error("budgets must be positive and finite; max-cycles must be nonnegative")
    if not 0 < args.target_occupancy <= 1:
        parser.error("target-occupancy must be in (0, 1]")
    run_loop(args.runs_dir, args.seconds, args.steps, args.max_cycles, args.target_occupancy)


if __name__ == "__main__":
    main()
