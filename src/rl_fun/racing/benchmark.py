"""Stored reference benchmark and helpers to compare a user's evolution with it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"


def load_benchmark(track: str) -> dict[str, Any]:
    """Load the stored reference benchmark for a track."""
    path = BENCHMARKS_DIR / f"{track}.json"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in BENCHMARKS_DIR.glob("*.json")))
        raise ValueError(f"unknown benchmark {track!r}; available benchmarks: {available}")
    return json.loads(path.read_text(encoding="utf-8"))


def _first_lap(history: list[dict[str, Any]]) -> int | None:
    for entry in history:
        if entry["finished"] > 0:
            return int(entry["generation"])
    return None


def compare(history: list[dict[str, Any]], benchmark: dict[str, Any]) -> str:
    """Russian text comparing the user's history with the reference."""
    reference = benchmark["history"]
    own_best = max(entry["best"] for entry in history)
    reference_best = max(entry["best"] for entry in reference)
    reference_lap = _first_lap(reference)
    own_lap = _first_lap(history)

    lines = [
        f"Твой лучший прогресс: {own_best:.2f} круга; у эталона: {reference_best:.2f}.",
    ]
    if reference_lap is not None:
        lines.append(f"Эталон прошёл круг на поколении {reference_lap + 1}.")
    if own_lap is not None:
        lines.append(f"Ты прошёл круг на поколении {own_lap + 1}.")
    else:
        lines.append("Ты пока не прошёл полный круг.")
    percent = 100.0 * own_best / reference_best if reference_best > 0 else 0.0
    lines.append(f"Ты на {percent:.0f}% от эталона.")
    return "\n".join(lines)


def plot_comparison(history: list[dict[str, Any]], benchmark: dict[str, Any]) -> Any:
    """Figure with best and mean progress of the user and of the reference per generation."""
    from matplotlib.figure import Figure

    reference = benchmark["history"]
    figure = Figure(figsize=(8, 4.5))
    axes = figure.subplots()
    for label, data, style in (
        ("Эталон: лучший", reference, {"color": "tab:green", "linestyle": "-"}),
        ("Эталон: средний", reference, {"color": "tab:green", "linestyle": "--"}),
        ("Твой: лучший", history, {"color": "tab:blue", "linestyle": "-"}),
        ("Твой: средний", history, {"color": "tab:blue", "linestyle": "--"}),
    ):
        key = "best" if "лучший" in label else "mean"
        axes.plot(
            [entry["generation"] + 1 for entry in data],
            [entry[key] for entry in data],
            label=label,
            **style,
        )
    axes.set_xlabel("Поколение")
    axes.set_ylabel("Прогресс (доля круга)")
    axes.set_title("Сравнение с эталоном")
    axes.grid(alpha=0.3)
    axes.legend()
    return figure
