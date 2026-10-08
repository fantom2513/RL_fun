from __future__ import annotations

import time
import traceback
import uuid
from pathlib import Path

import gymnasium as gym
import numpy as np

from rl_fun.algorithms.registry import get_algorithm
from rl_fun.environments.factory import make_environment
from rl_fun.experiments.config import ExperimentConfig
from rl_fun.experiments.result import RunSummary
from rl_fun.tracking.artifacts import collect_metadata, create_run_dir, write_json_atomic
from rl_fun.tracking.metrics import JsonlMetricSink


def run_once(
    config: ExperimentConfig,
    seed: int,
    run_id: str | None = None,
) -> RunSummary:
    config.validate()
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in config.run.seeds:
        raise ValueError(f"seed {seed!r} is not present in the experiment configuration")
    algorithm = get_algorithm(config.algorithm.id)
    identifier = run_id if run_id is not None else f"seed-{seed}-{uuid.uuid4().hex[:12]}"
    run_dir = create_run_dir(Path(config.run.output_root), config.name, identifier)
    write_json_atomic(run_dir / "config.json", config.to_dict())
    write_json_atomic(run_dir / "metadata.json", collect_metadata(seed))
    started = time.perf_counter()
    env: gym.Env | None = None
    error: str | None = None
    interruption: KeyboardInterrupt | None = None
    try:
        env = make_environment(config.environment)
        rng = np.random.default_rng(seed)
        algorithm.validate(env, config.environment.id)
        with JsonlMetricSink(run_dir / "metrics.jsonl") as metrics:
            result = algorithm.run(
                env, config.run.total_steps, seed, rng, metrics, config.algorithm.kwargs,
            )
        summary = RunSummary(
            status="success", seed=seed, run_dir=run_dir, metrics=result.metrics,
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception:
        error = traceback.format_exc()
    except KeyboardInterrupt as interrupt:
        interruption = interrupt
        error = traceback.format_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                cleanup_error = traceback.format_exc()
                error = (
                    f"{error}\nEnvironment cleanup failed:\n{cleanup_error}"
                    if error else cleanup_error
                )
    if error is not None:
        (run_dir / "error.txt").write_text(error, encoding="utf-8")
        summary = RunSummary(
            status="cancelled" if interruption is not None else "failure",
            seed=seed, run_dir=run_dir, metrics={},
            elapsed_seconds=time.perf_counter() - started, error=error,
        )
    write_json_atomic(run_dir / "summary.json", summary.to_dict())
    if interruption is not None:
        raise interruption
    return summary
