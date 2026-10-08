"""Сгенерировать зафиксированный бенчмарк эталонной эволюции."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from rl_fun.racing import reference
from rl_fun.racing.benchmark import BENCHMARKS_DIR

FLEET_KWARGS = {"max_steps": 1500}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сгенерировать бенчмарк эталона")
    parser.add_argument("--track", default="oval")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=BENCHMARKS_DIR)
    arguments = parser.parse_args(argv)

    run = reference.train(
        arguments.track,
        seed=arguments.seed,
        generations=arguments.generations,
        params=reference.EvolutionParams(),
        fleet_kwargs=FLEET_KWARGS,
    )
    for entry in run.history:
        print(
            f"поколение {entry['generation'] + 1:3d}: лучший {entry['best']:.3f}, "
            f"средний {entry['mean']:.3f}, финишировало {entry['finished']}"
        )

    params = asdict(run.params)
    params["hidden"] = list(params["hidden"])
    payload = {
        "track": run.track,
        "seed": run.seed,
        "params": params,
        "fleet": FLEET_KWARGS,
        "history": run.history,
        "best_weights": [float(w) for w in run.best_weights],
        "sizes": run.sizes,
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    path = arguments.output_dir / f"{run.track}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Записано: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
