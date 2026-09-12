import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def read_curve(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    events = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    steps = np.asarray([event["step"] for event in events], dtype=np.int64)
    rewards = np.asarray(
        [event["metrics"]["cumulative_reward"] for event in events],
        dtype=np.float64,
    )
    return steps, rewards


def common_curves(run_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    curves = [read_curve(path) for path in run_dirs]
    common_steps = set(curves[0][0].tolist())
    for steps, _ in curves[1:]:
        common_steps.intersection_update(steps.tolist())
    ordered = np.asarray(sorted(common_steps), dtype=np.int64)
    matrix = np.asarray(
        [
            [rewards[np.flatnonzero(steps == step)[0]] for step in ordered]
            for steps, rewards in curves
        ],
        dtype=np.float64,
    )
    return ordered, matrix


def write_comparison(output_dir: Path, steps: np.ndarray, matrix: np.ndarray) -> None:
    means = np.mean(matrix, axis=0)
    standard_deviations = np.std(matrix, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("step", "mean_cumulative_reward", "std_cumulative_reward"),
        )
        writer.writeheader()
        writer.writerows(
            {
                "step": int(step),
                "mean_cumulative_reward": float(mean),
                "std_cumulative_reward": float(standard_deviation),
            }
            for step, mean, standard_deviation in zip(
                steps,
                means,
                standard_deviations,
                strict=True,
            )
        )
    figure, axes = plt.subplots()
    axes.plot(steps, means, label="mean cumulative reward")
    axes.fill_between(steps, means - standard_deviations, means + standard_deviations, alpha=0.2)
    axes.set_xlabel("step")
    axes.set_ylabel("cumulative reward")
    axes.legend()
    figure.savefig(output_dir / "comparison.png", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare RL Fun experiment runs")
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("comparison"))
    arguments = parser.parse_args()
    try:
        steps, matrix = common_curves(arguments.run_dirs)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, IndexError) as error:
        parser.error(str(error))
    if not len(steps):
        return 2
    write_comparison(arguments.output_dir, steps, matrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
