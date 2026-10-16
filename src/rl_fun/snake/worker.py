"""Spawn-safe training entry point and cooperative controls."""

import json
import time
from pathlib import Path


def run_worker(config_data, connection, run_dir, resume_path=None, visualization=None):
    from rl_fun.snake.checkpoint import load_checkpoint, save_checkpoint
    from rl_fun.snake.config import SnakeConfig
    from rl_fun.snake.trainers import create_trainer

    trainer = None
    paused = stopped = False
    run_dir = Path(run_dir)
    visualization = visualization or {"enabled": False, "fps": 8}

    def send(kind, **data):
        connection.send({"kind": kind, **data})

    def save():
        save_checkpoint(run_dir / "model.pt", trainer.state_dict())
        send("saved", iteration=trainer.completed)

    def control():
        nonlocal paused, stopped, visualization
        while True:
            while connection.poll():
                command = connection.recv()
                if isinstance(command, dict) and command.get("command") == "visualization":
                    visualization = command["settings"]
                elif command == "stop":
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

    skipped = 0

    def should_emit(ended=False):
        nonlocal skipped
        if not visualization["enabled"] or stopped:
            return False
        # Above 60 fps the pipe and the browser cannot keep up: show every n-th frame.
        stride = max(1, visualization["fps"] // 60)
        skipped = (skipped + 1) % stride
        return ended or skipped == 0

    def frame(row):
        stride = max(1, visualization["fps"] // 60)
        send("frame", frame=row)
        started = time.monotonic()
        while (
            visualization["enabled"] and time.monotonic() - started < stride / visualization["fps"]
        ):
            if not control():
                return
            time.sleep(0.01)

    def emit(metric):
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metric, ensure_ascii=False) + "\n")
        if (
            trainer.config.algorithm == "evolution"
            or trainer.completed == 1
            or trainer.completed % 20 == 0
        ):
            save()
        send("metric", metric=metric)

    frame.enabled = lambda: visualization["enabled"] and not stopped
    frame.should_emit = should_emit

    try:
        config = SnakeConfig.from_dict(config_data)
        restored = load_checkpoint(resume_path) if resume_path else None
        if restored:
            config_data = config.to_dict()
            budget = "generations" if config.algorithm == "evolution" else "episodes"
            config_data[budget] += restored["completed"]
            config = SnakeConfig.from_dict(config_data)
        trainer = create_trainer(config)
        if restored:
            trainer.load_state_dict(restored)
        send("status", status="running")
        trainer.run(control, emit, frame)
        save()
        send("status", status="stopped" if stopped else "finished")
    except (BrokenPipeError, EOFError, OSError):
        pass
    except Exception:
        send("status", status="error", error="ошибка обучения; проверьте параметры и модель")
    finally:
        connection.close()
