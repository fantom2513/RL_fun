"""Manual quality check of the PPO learner: train with the lab defaults and print learning curves.

Not a test. Example:

    uv run python scripts/ppo_check.py --track oval --minutes 5
    uv run python scripts/ppo_check.py --track circuit --minutes 4 --learning-rate 1e-3

Every iteration prints the episodes that ended in it (count, mean and best progress in laps,
finished laps), the mean episode return, entropy, policy std, KL and the wall time. At the end the
deterministic policy (mean action, no noise) of the last iteration and of the best snapshot drives
one car for a full episode.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import fields
from typing import Any

import numpy as np

from rl_fun.lab.config import PPOParams, RunConfig
from rl_fun.learners.ppo import PPOLearner
from rl_fun.racing import reference
from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.generation import run_generation

PPO_OPTIONS = {item.name: item.type for item in fields(PPOParams)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--track", default="oval")
    parser.add_argument("--iterations", type=int, default=10_000, help="upper bound")
    parser.add_argument("--minutes", type=float, default=5.0, help="wall-time budget")
    parser.add_argument("--population", type=int, default=RunConfig().population)
    parser.add_argument("--max-steps", type=int, default=RunConfig().max_steps)
    parser.add_argument("--seed", type=int, default=RunConfig().seed)
    parser.add_argument("--every", type=int, default=1, help="print every N-th iteration")
    for name in PPO_OPTIONS:
        number = int if name in ("rollout_steps", "epochs", "minibatch_size") else float
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=number, default=None)
    return parser.parse_args()


def evaluate(config: RunConfig, weights: np.ndarray) -> tuple[float, bool, float]:
    """Drive one car with the deterministic policy; returns (progress, finished, lap steps)."""
    fleet = RacingFleet(
        1, track=config.track, model=config.model, max_steps=config.max_steps,
        ray_range=config.ray_range,
    )
    sizes, activation = config.model.layer_sizes, config.model.activation
    result = run_generation(
        fleet,
        weights[None, :],
        lambda w, o: reference.forward(w, o, sizes, activation),
    )
    return float(result.progress[0]), bool(result.finished[0]), float(result.lap_steps[0])


def main() -> None:
    args = parse_args()
    overrides: dict[str, Any] = {
        name: getattr(args, name) for name in PPO_OPTIONS if getattr(args, name) is not None
    }
    config = RunConfig(
        name="ppo-check",
        track=args.track,
        population=args.population,
        max_steps=args.max_steps,
        seed=args.seed,
        learner="ppo",
        ppo=PPOParams(**overrides),
    )
    print(f"track={config.track} population={config.population} max_steps={config.max_steps}")
    print("ppo:", config.ppo.to_dict(), flush=True)

    fleet = RacingFleet(
        config.population, track=config.track, model=config.model,
        max_steps=config.max_steps, ray_range=config.ray_range,
    )
    learner = PPOLearner()
    learner.setup(config, fleet, np.random.default_rng(config.seed), None)

    print(
        f"{'it':>5} {'time':>7} {'eps':>4} {'mean':>6} {'best':>6} {'fin':>4} {'fin_tot':>7} "
        f"{'lap':>5} {'return':>7} {'entropy':>8} {'std':>6} {'kl':>7}",
        flush=True,
    )
    started = time.perf_counter()
    total_finished = 0
    first_finish: int | None = None
    rows: list[tuple[int, float, float, int]] = []
    for iteration in range(args.iterations):
        result = learner.run_iteration(iteration)
        elapsed = time.perf_counter() - started
        extra = result.extra
        episodes = int(extra["episodes"])
        finished = result.finished if episodes else 0
        total_finished += finished
        if finished and first_finish is None:
            first_finish = iteration
        if episodes:
            rows.append((iteration, elapsed, result.mean, finished))
        lap = "-" if result.best_lap_steps is None else str(result.best_lap_steps)
        if iteration % args.every == 0 or finished:
            print(
                f"{iteration:5d} {elapsed:7.1f} {episodes:4d} "
                f"{result.mean:6.3f} {result.best:6.3f} "
                f"{finished:4d} {total_finished:7d} {lap:>5} {extra['mean_return']:7.2f} "
                f"{extra['entropy']:8.3f} {extra['std']:6.3f} {extra['kl']:7.4f}",
                flush=True,
            )
        if elapsed > args.minutes * 60:
            break

    elapsed = time.perf_counter() - started
    print(f"\ndone: {iteration + 1} iterations in {elapsed:.1f} s, finished laps {total_finished}")
    print(f"first finished lap at iteration: {first_finish}")
    if rows:
        tail = rows[-max(1, len(rows) // 10) :]
        print(f"mean progress of the last {len(tail)} reporting iterations: "
              f"{np.mean([row[2] for row in tail]):.3f}")
    final = evaluate(config, learner._agent.export())
    print(f"deterministic final policy: progress {final[0]:.3f}, finished {final[1]}, "
          f"lap steps {final[2]}")
    snapshot = learner.best_snapshot()
    if snapshot is not None:
        best = evaluate(config, np.asarray(snapshot["weights"]))
        print(f"deterministic best snapshot (iteration {snapshot['generation']}): progress "
              f"{best[0]:.3f}, finished {best[1]}, lap steps {best[2]}", flush=True)


if __name__ == "__main__":
    main()
