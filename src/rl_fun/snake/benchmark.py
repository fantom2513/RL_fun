"""Bounded training and an independent, reproducible completion benchmark."""

import copy
import hashlib
import json
import platform
import subprocess
import time
import uuid
from pathlib import Path

import torch

from rl_fun.snake.checkpoint import load_checkpoint, playback_weights, save_checkpoint
from rl_fun.snake.policy import Policy, evaluate, evaluation_rank
from rl_fun.snake.presets import benchmark_seeds, verification_status
from rl_fun.snake.trainers import create_trainer


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def source_metadata():
    def git(*arguments):
        result = subprocess.run(["git", *arguments], capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
    }


def run_benchmark(
    config,
    runs_dir,
    train_seconds=60,
    resume=None,
    progress=print,
    validate=False,
    train_steps=None,
    stop_requested=None,
):
    if train_steps is not None and (type(train_steps) is not int or train_steps <= 0):
        raise ValueError("бюджет ходов должен быть положительным целым числом")
    if config.algorithm == "evolution":
        raise ValueError("проверка обучения требует алгоритм с бюджетом эпизодов")
    if config.observation_mode not in ("board", "egocentric"):
        raise ValueError("проверка прохождения требует наблюдения board или egocentric")
    if validate and not resume:
        raise ValueError("финальная проверка требует замороженного checkpoint")
    restored = load_checkpoint(resume) if resume else None
    if validate and config.to_dict() != type(config).from_dict(restored["config"]).to_dict():
        raise ValueError("параметры финальной проверки должны совпадать с моделью")
    directory = Path(runs_dir).resolve() / "snake" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    started = time.monotonic()
    created = time.time()
    trainer = create_trainer(config)
    if restored:
        trainer.load_state_dict(restored)
    tuning_seeds, validation_seeds = benchmark_seeds()
    history = []
    train_started = time.monotonic()
    initial_steps = trainer.steps

    def emit(row):
        history.append(row)
        with (directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if row["iteration"] == 1 or row["iteration"] % 200 == 0:
            progress(
                f"episode={row['iteration']} steps={trainer.steps} "
                f"score={row['mean_score']:.2f} "
                f"eval_fill={row['evaluation_occupancy']:.3f} "
                f"wins={row['evaluation_win_rate']:.3f}"
            )

    if not validate:

        def control():
            if stop_requested is not None and stop_requested():
                return False
            within_steps = (
                train_steps is None
                or trainer.steps - initial_steps < train_steps
                or getattr(trainer, "optimization", None) is not None
            )
            return time.monotonic() - train_started < train_seconds and within_steps

        trainer.run(control, emit)
    training_elapsed = time.monotonic() - train_started
    state = trainer.state_dict()
    model = Policy(config.size, config.observation_mode)
    model.load_state_dict(playback_weights(state))
    selected_weights = copy.deepcopy(model.state_dict())
    model.action_adapter = state.get("action_adapter")
    progress(f"Training stopped at episode={trainer.completed}; evaluating frozen best policy.")
    tuning = evaluate(
        model, config.size, tuning_seeds, config.max_steps, env_options=config.env_options()
    )
    latest = None
    if not validate:
        model.load_state_dict(state["model"])
        latest = evaluate(
            model, config.size, tuning_seeds, config.max_steps, env_options=config.env_options()
        )
        state["last_evaluation"] = latest
        if evaluation_rank(latest) > evaluation_rank(tuning):
            tuning = latest
            selected_weights = copy.deepcopy(state["model"])
        state["best_model"] = selected_weights
        state["best_rank"] = list(evaluation_rank(tuning))
        model.load_state_dict(playback_weights(state))
    save_checkpoint(directory / "model.pt", state)
    result = (
        evaluate(
            model, config.size, validation_seeds, config.max_steps, env_options=config.env_options()
        )
        if validate
        else tuning
    )
    checkpoint = directory / "model.pt"
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {key: value for key, value in result.items() if key != "frame"}
    report.update(
        {
            "status": verification_status(config.size, result["episodes"], result["wins"])
            if validate
            else "experimental",
            "phase": "validation" if validate else "tuning",
            "config": config.to_dict(),
            "training_seed": trainer.training_seed,
            "resumed_from": str(Path(resume).parent.name) if resume else None,
            "training_seed_range": [0, 999999],
            "tuning_seeds": tuning_seeds,
            "validation_seeds": validation_seeds if validate else [],
            "tuning": {k: v for k, v in tuning.items() if k != "frame"},
            "model_sha256": digest,
            "model_mtime_ns": checkpoint.stat().st_mtime_ns,
            "elapsed_seconds": time.monotonic() - started,
            "training_seconds": training_elapsed,
            "training_steps_budget": train_steps,
            "steps": trainer.steps,
            "steps_added": trainer.steps - initial_steps,
            "completed": trainer.completed,
            "updates": trainer.updates,
            "pending_rollout_steps": len(getattr(trainer, "rollout", [])),
            "latest_tuning": {k: v for k, v in latest.items() if k != "frame"} if latest else None,
            "source": source_metadata(),
            "method": "reinforcement_learning_from_scratch",
        }
    )
    write_json(directory / ("validation.json" if validate else "evaluation.json"), report)
    write_json(
        directory / "record.json",
        {
            "id": directory.name,
            "created": created,
            "config": config.to_dict(),
            "status": "finished" if trainer.completed >= config.episodes else "stopped",
            "history": history[-500:],
            "saved": True,
            "saved_iteration": trainer.completed,
            "verification": {
                key: report[key]
                for key in (
                    "status",
                    "wins",
                    "episodes",
                    "win_rate",
                    "occupancy",
                    "method",
                    "model_sha256",
                    "model_mtime_ns",
                )
            }
            if validate
            else None,
        },
    )
    progress(
        f"{report['phase']}: {report['wins']}/{report['episodes']} wins; "
        f"fill={report['occupancy']:.3f}; status={report['status']}; directory={directory}"
    )
    return directory
