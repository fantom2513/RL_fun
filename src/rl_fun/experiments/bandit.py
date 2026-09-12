from __future__ import annotations

import time
import traceback
import uuid
from pathlib import Path

import numpy as np

from rl_fun.algorithms.bandits.epsilon_greedy import run_epsilon_greedy
from rl_fun.algorithms.bandits.greedy import run_greedy
from rl_fun.algorithms.bandits.random_agent import run_random
from rl_fun.environments.bandit import StationaryBanditEnv
from rl_fun.experiments.config import ExperimentConfig
from rl_fun.experiments.result import RunSummary
from rl_fun.tracking.artifacts import collect_metadata, create_run_dir, write_json_atomic
from rl_fun.tracking.metrics import JsonlMetricSink


def run_bandit_once(
    config: ExperimentConfig,
    seed: int,
    run_id: str | None = None,
) -> RunSummary:
    config.validate()
    if seed not in config.seeds:
        raise ValueError(f"seed {seed} is not present in the experiment configuration")
    identifier = run_id or f"seed-{seed}-{uuid.uuid4().hex[:12]}"
    run_dir = create_run_dir(Path(config.output_root), config.name, identifier)
    write_json_atomic(run_dir / "config.json", config.to_dict())
    write_json_atomic(run_dir / "metadata.json", collect_metadata(seed))
    started = time.perf_counter()
    env: StationaryBanditEnv | None = None
    try:
        arms = int(config.parameters.get("arms", 10))
        reward_std = float(config.parameters.get("reward_std", 1.0))
        env = StationaryBanditEnv(arms=arms, horizon=config.steps, reward_std=reward_std)
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        with JsonlMetricSink(run_dir / "metrics.jsonl") as metrics:
            match config.algorithm:
                case "random":
                    outcome = run_random(env, config.steps, rng, metrics)
                case "greedy":
                    outcome = run_greedy(env, config.steps, rng, metrics)
                case "epsilon_greedy":
                    epsilon = float(config.parameters.get("epsilon", 0.1))
                    outcome = run_epsilon_greedy(
                        env,
                        config.steps,
                        rng,
                        metrics,
                        epsilon=epsilon,
                    )
                case unknown:
                    raise ValueError(f"unsupported bandit algorithm: {unknown}")
        summary = RunSummary(
            status="success",
            seed=seed,
            run_dir=run_dir,
            cumulative_reward=outcome.cumulative_reward,
            cumulative_regret=outcome.cumulative_regret,
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception:
        error = traceback.format_exc()
        (run_dir / "error.txt").write_text(error, encoding="utf-8")
        summary = RunSummary(
            status="failure",
            seed=seed,
            run_dir=run_dir,
            cumulative_reward=None,
            cumulative_regret=None,
            elapsed_seconds=time.perf_counter() - started,
            error=error,
        )
    finally:
        if env is not None:
            env.close()
    write_json_atomic(run_dir / "summary.json", summary.to_dict())
    return summary
