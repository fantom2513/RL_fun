import json

from rl_fun.experiments.bandit import run_bandit_once
from rl_fun.experiments.config import ExperimentConfig


def test_one_run_writes_reproducible_artifacts(tmp_path):
    # Arrange
    config = ExperimentConfig(
        name="smoke-bandit",
        algorithm="epsilon_greedy",
        seeds=(17,),
        steps=10,
        workers=1,
        output_root=str(tmp_path),
        parameters={"arms": 3, "epsilon": 0.1, "reward_std": 0.0},
    )

    # Act
    summary = run_bandit_once(config, seed=17, run_id="fixed-run")

    # Assert
    assert json.loads((summary.run_dir / "summary.json").read_text(encoding="utf-8"))["status"] == "success"


def test_same_seed_produces_same_numeric_summary(tmp_path):
    # Arrange
    config = ExperimentConfig(
        name="repeatable",
        algorithm="greedy",
        seeds=(9,),
        steps=20,
        workers=1,
        output_root=str(tmp_path),
        parameters={"arms": 4, "reward_std": 0.0},
    )

    # Act
    first = run_bandit_once(config, seed=9, run_id="first")
    second = run_bandit_once(config, seed=9, run_id="second")

    # Assert
    assert (first.cumulative_reward, first.cumulative_regret) == (
        second.cumulative_reward,
        second.cumulative_regret,
    )
