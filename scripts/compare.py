import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def read_curve(run_dir: Path, metric: str) -> tuple[np.ndarray, np.ndarray]:
    events = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = [event for event in events if metric in event["metrics"]]
    steps = np.asarray([event["step"] for event in events], dtype=np.int64)
    rewards = np.asarray(
        [event["metrics"][metric] for event in events],
        dtype=np.float64,
    )
    return steps, rewards


def common_curves(run_dirs: list[Path], metric: str) -> tuple[np.ndarray, np.ndarray]:
    curves = []
    for path in run_dirs:
        summary_path = path / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary["status"] != "success":
                continue
        steps, values = read_curve(path, metric)
        if len(steps):
            curves.append((steps, values))
    if not curves:
        raise ValueError(f"Ни один успешный запуск не содержит метрику {metric!r}")
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


def write_comparison(
    output_dir: Path, steps: np.ndarray, matrix: np.ndarray, metric: str,
) -> None:
    means = np.mean(matrix, axis=0)
    standard_deviations = np.std(matrix, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("step", "mean", "std"),
        )
        writer.writeheader()
        writer.writerows(
            {
                "step": int(step),
                "mean": float(mean),
                "std": float(standard_deviation),
            }
            for step, mean, standard_deviation in zip(
                steps,
                means,
                standard_deviations,
                strict=True,
            )
        )
    figure, axes = plt.subplots()
    axes.plot(steps, means, label=metric)
    axes.fill_between(
        steps, means - standard_deviations, means + standard_deviations,
        alpha=0.2, label="Среднее ± стандартное отклонение",
    )
    axes.set_title("Сравнение запусков")
    axes.set_xlabel("Шаг")
    axes.set_ylabel(f"Значение метрики: {metric}")
    axes.legend()
    figure.savefig(output_dir / "comparison.png", bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сравнить запуски RL Fun по выбранной метрике")
    parser.add_argument("run_dirs", type=Path, nargs="+", help="Директории артефактов запусков")
    parser.add_argument("--metric", required=True, help="Ключ сравниваемой метрики")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("comparison"), help="Директория CSV и графика",
    )
    arguments = parser.parse_args(argv)
    try:
        steps, matrix = common_curves(arguments.run_dirs, arguments.metric)
        if not len(steps):
            raise ValueError("Нет общих шагов для выбранной метрики")
        write_comparison(arguments.output_dir, steps, matrix, arguments.metric)
    except KeyboardInterrupt:
        return 130
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, IndexError) as error:
        parser.error(f"Не удалось сравнить запуски: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
