"""Spawn-safe training entry point and cooperative controls."""

import json
import time
from pathlib import Path


def run_worker(config_data, connection, run_dir, resume_path=None):
    from rl_fun.snake.checkpoint import load_checkpoint, save_checkpoint
    from rl_fun.snake.config import SnakeConfig
    from rl_fun.snake.dqn import DQNTrainer
    from rl_fun.snake.evolution import EvolutionTrainer

    trainer = None
    paused = stopped = False
    run_dir = Path(run_dir)

    def send(kind, **data):
        connection.send({"kind": kind, **data})

    def save():
        save_checkpoint(run_dir / "model.pt", trainer.state_dict())
        send("saved", iteration=trainer.completed)

    def control():
        nonlocal paused, stopped
        while True:
            while connection.poll():
                command = connection.recv()
                if command == "stop":
                    stopped = True
                    paused = False
                elif command == "pause":
                    paused = True
                    send("status", status="paused")
                elif command == "resume":
                    paused = False
                    send("status", status="running")
                elif command == "save":
                    save()
            if stopped or not paused:
                return not stopped
            time.sleep(0.04)

    def emit(metric):
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metric, ensure_ascii=False) + "\n")
        if trainer.completed == 1 or trainer.completed % 20 == 0:
            save()
        send("metric", metric=metric)

    try:
        config = SnakeConfig.from_dict(config_data)
        restored = load_checkpoint(resume_path) if resume_path else None
        if restored:
            config_data = config.to_dict()
            budget = "episodes" if config.algorithm == "dqn" else "generations"
            config_data[budget] += restored["completed"]
            config = SnakeConfig.from_dict(config_data)
        trainer = DQNTrainer(config) if config.algorithm == "dqn" else EvolutionTrainer(config)
        if restored:
            trainer.load_state_dict(restored)
        send("status", status="running")
        trainer.run(control, emit)
        save()
        send("status", status="stopped" if stopped else "finished")
    except (BrokenPipeError, EOFError, OSError):
        pass
    except Exception:
        send("status", status="error", error="ошибка обучения; проверьте параметры и модель")
    finally:
        connection.close()
