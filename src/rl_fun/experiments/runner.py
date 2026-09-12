from __future__ import annotations

import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass

from rl_fun.experiments.bandit import run_bandit_once
from rl_fun.experiments.config import ExperimentConfig
from rl_fun.experiments.result import RunSummary


@dataclass(frozen=True, slots=True)
class BatchSummary:
    runs: tuple[RunSummary, ...]

    @property
    def success_count(self) -> int:
        return sum(run.status == "success" for run in self.runs)

    @property
    def failure_count(self) -> int:
        return sum(run.status == "failure" for run in self.runs)


def _run_seed(config_dict: dict[str, object], seed: int) -> RunSummary:
    config = ExperimentConfig.from_dict(config_dict)
    return run_bandit_once(config, seed)


def run_many(config: ExperimentConfig) -> BatchSummary:
    config.validate()
    if config.workers == 1:
        runs = tuple(run_bandit_once(config, seed) for seed in config.seeds)
    else:
        context = multiprocessing.get_context("spawn")
        futures: list[Future[RunSummary]] = []
        with ProcessPoolExecutor(
            max_workers=min(config.workers, len(config.seeds)),
            mp_context=context,
        ) as pool:
            try:
                futures = [
                    pool.submit(_run_seed, config.to_dict(), seed)
                    for seed in config.seeds
                ]
                runs = tuple(future.result() for future in futures)
            except KeyboardInterrupt:
                for future in futures:
                    future.cancel()
                raise
    return BatchSummary(tuple(sorted(runs, key=lambda run: run.seed)))
