import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from rl_fun.racing import reference as ref
from rl_fun.racing.benchmark import compare, load_benchmark, plot_comparison
from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.generation import run_generation

HISTORY = [
    {"generation": 0, "best": 0.10, "mean": 0.05, "finished": 0},
    {"generation": 1, "best": 0.30, "mean": 0.12, "finished": 0},
]


def test_oval_benchmark_has_expected_fields():
    benchmark = load_benchmark("oval")

    for key in ("track", "seed", "params", "fleet", "history", "best_weights", "sizes"):
        assert key in benchmark
    assert benchmark["track"] == "oval"


def test_reference_completes_a_lap_in_benchmark():
    history = load_benchmark("oval")["history"]

    assert max(entry["best"] for entry in history) >= 1.0
    assert any(entry["finished"] > 0 for entry in history)


def test_benchmark_is_reproducible():
    benchmark = load_benchmark("oval")
    params = ref.EvolutionParams(**{**benchmark["params"], "hidden": tuple(benchmark["params"]["hidden"])})

    run = ref.train(
        benchmark["track"], seed=benchmark["seed"], generations=3,
        params=params, fleet_kwargs=benchmark["fleet"],
    )

    for new, stored in zip(run.history, benchmark["history"][:3]):
        assert new["best"] == pytest.approx(stored["best"])
        assert new["mean"] == pytest.approx(stored["mean"])


def test_stored_best_weights_still_drive_a_lap():
    benchmark = load_benchmark("oval")
    sizes = benchmark["sizes"]
    weights = np.array(benchmark["best_weights"])
    fleet = RacingFleet(1, track=benchmark["track"], **benchmark["fleet"])

    result = run_generation(
        fleet, weights[None, :], lambda w, o: ref.forward(w, o, sizes)
    )

    assert result.finished[0]


def test_compare_mentions_both_sides():
    text = compare(HISTORY, load_benchmark("oval"))

    assert "эталон" in text.lower()
    assert "0.30" in text


def test_compare_reports_lap_completion():
    history = [*HISTORY, {"generation": 2, "best": 1.0, "mean": 0.4, "finished": 3}]

    assert "круг" in compare(history, load_benchmark("oval")).lower()


def test_unknown_benchmark_raises():
    with pytest.raises(ValueError, match="benchmark"):
        load_benchmark("nope")


def test_plot_comparison_returns_figure():
    figure = plot_comparison(HISTORY, load_benchmark("oval"))

    assert figure.axes
