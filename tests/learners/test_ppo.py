import math
import subprocess
import sys

import numpy as np
import pytest

from rl_fun.lab.config import PPOParams, RunConfig
from rl_fun.learners.base import IterationResult
from rl_fun.learners.ppo import PPOLearner, compute_gae
from rl_fun.learners.registry import get_learner
from rl_fun.learners.schemas import PPO_SCHEMA
from rl_fun.racing import reference
from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.model_spec import ModelSpec

EXTRA_KEYS = {"policy_loss", "value_loss", "entropy", "std", "kl", "mean_return", "episodes"}


def _config(**overrides) -> RunConfig:
    values = {
        "track": "oval",
        "population": 6,
        "max_steps": 50,
        "seed": 5,
        "learner": "ppo",
        "ppo": {"rollout_steps": 16, "minibatch_size": 32, "epochs": 2},
    }
    values.update(overrides)
    return RunConfig.from_dict(values)


def _fleet(config: RunConfig) -> RacingFleet:
    return RacingFleet(
        config.population,
        track=config.track,
        model=config.model,
        max_steps=config.max_steps,
        ray_range=config.ray_range,
    )


def _learner(config: RunConfig, view=None) -> tuple[PPOLearner, RacingFleet]:
    fleet = _fleet(config)
    learner = PPOLearner()
    learner.setup(config, fleet, np.random.default_rng(config.seed), view)
    return learner, fleet


# ---- GAE -------------------------------------------------------------------------------------


def test_gae_matches_a_hand_computed_example_with_terminal_and_time_limit():
    # Arrange: car 0 runs on (bootstrap from the value after the rollout); car 1 crashes at t=1
    # (terminal, no bootstrap) and is cut by the time limit at t=2 (bootstrap from next_values).
    rewards = np.array([[1.0, 0.0], [1.0, -1.0], [1.0, 2.0]])
    values = np.array([[0.5, 1.0], [0.5, 1.0], [0.5, 1.0]])
    next_values = np.array([[0.5, 1.0], [0.5, 7.0], [2.0, 3.0]])
    terminals = np.array([[False, False], [False, True], [False, False]])
    dones = np.array([[False, False], [False, True], [False, True]])

    # Act
    advantages, returns = compute_gae(
        rewards, values, next_values, terminals, dones, gamma=0.9, lam=0.5
    )

    # Assert
    expected = np.array([[1.84325, -1.0], [1.985, -2.0], [2.3, 3.7]])
    assert advantages == pytest.approx(expected)
    assert returns == pytest.approx(expected + values)


def test_gae_with_lambda_one_and_no_done_is_the_discounted_return_minus_value():
    # Arrange
    rewards = np.array([[1.0], [2.0], [3.0]])
    values = np.zeros((3, 1))
    next_values = np.array([[0.0], [0.0], [10.0]])
    flags = np.zeros((3, 1), dtype=bool)

    # Act
    _, returns = compute_gae(rewards, values, next_values, flags, flags, gamma=0.5, lam=1.0)

    # Assert
    assert returns[:, 0] == pytest.approx([1 + 0.5 * 2 + 0.25 * 3 + 0.125 * 10, 2 + 1.5 + 2.5, 8.0])


# ---- losses and the update -------------------------------------------------------------------


def test_clipped_loss_has_known_value_and_zero_gradient_outside_the_clip_zone():
    # Arrange
    import torch

    from rl_fun.learners.ppo_core import clipped_policy_loss

    ratios = torch.tensor([1.5, 1.1, 0.5, 0.9, 0.7], dtype=torch.float64)
    new_log_probs = torch.log(ratios).requires_grad_(True)
    old_log_probs = torch.zeros(5, dtype=torch.float64)
    advantages = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0], dtype=torch.float64)

    # Act
    loss = clipped_policy_loss(new_log_probs, old_log_probs, advantages, 0.2)
    loss.backward()

    # Assert
    assert loss.item() == pytest.approx(-0.26)
    gradient = new_log_probs.grad.numpy()
    assert gradient == pytest.approx([0.0, -1.1 / 5, 0.0, 0.9 / 5, -0.7 / 5])


def test_policy_output_has_the_model_shape_bounds_and_initial_std():
    # Arrange
    import torch

    from rl_fun.learners.ppo_core import PolicyNet

    model = ModelSpec()
    policy = PolicyNet(model.layer_sizes, model.activation, 0.6, np.random.default_rng(0))
    observations = torch.from_numpy(np.random.default_rng(1).normal(0, 3, (64, 8))).float()

    # Act
    mean = policy.mean(observations)
    log_prob = policy.log_prob(observations, mean)

    # Assert
    assert tuple(mean.shape) == (64, 2)
    assert float(mean.abs().max()) <= 1.0
    assert tuple(log_prob.shape) == (64,)
    assert policy.std().detach().numpy() == pytest.approx([0.6, 0.6])


@pytest.mark.parametrize("activation", ["tanh", "relu", "sigmoid"])
def test_exported_weights_reproduce_the_policy_mean_with_reference_forward(activation):
    # Arrange
    import torch

    from rl_fun.learners.ppo_core import PolicyNet, export_weights

    model = ModelSpec(hidden=(7, 4), activation=activation)
    policy = PolicyNet(model.layer_sizes, activation, 0.5, np.random.default_rng(2))
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=torch.Generator().manual_seed(3)))
    observations = np.random.default_rng(4).uniform(-1, 1, (10, 8))

    # Act
    weights = export_weights(policy)
    expected = policy.mean(torch.from_numpy(observations).float()).detach().numpy()

    # Assert
    assert weights.shape == (model.weight_count,)
    for row, observation in enumerate(observations):
        actual = reference.forward(weights, observation, model.layer_sizes, activation)
        assert actual == pytest.approx(expected[row], abs=1e-5)


def test_update_moves_the_mean_action_towards_positive_advantages():
    # Arrange
    import torch

    from rl_fun.learners.ppo_core import PolicyNet, ValueNet, ppo_update

    rng = np.random.default_rng(0)
    model = ModelSpec()
    policy = PolicyNet(model.layer_sizes, model.activation, 0.5, rng)
    value = ValueNet(model.layer_sizes, model.activation, rng)
    optimizer = torch.optim.Adam([*policy.parameters(), *value.parameters()], lr=3e-3)
    observations = rng.uniform(0, 1, (512, 8)).astype(np.float32)
    obs_tensor = torch.from_numpy(observations)
    with torch.no_grad():
        mean = policy.mean(obs_tensor)
        actions = mean + policy.std() * torch.randn(mean.shape, generator=torch.Generator())
        log_probs = policy.log_prob(obs_tensor, actions)
    steer_before = float(mean[:, 0].mean())
    batch = {
        "observations": observations,
        "actions": actions.numpy(),
        "log_probs": log_probs.numpy(),
        "advantages": (actions[:, 0] - mean[:, 0]).numpy(),
        "returns": np.zeros(512, dtype=np.float32),
    }
    params = PPOParams(learning_rate=3e-3, epochs=4, minibatch_size=64, entropy_coef=0.0)

    # Act
    metrics = ppo_update(policy, value, optimizer, batch, params, rng)

    # Assert
    with torch.no_grad():
        steer_after = float(policy.mean(obs_tensor)[:, 0].mean())
    assert steer_after > steer_before + 0.05
    assert set(metrics) == {"policy_loss", "value_loss", "entropy", "kl"}
    assert all(math.isfinite(value) for value in metrics.values())


# ---- the learner -----------------------------------------------------------------------------


def test_tiny_iteration_runs_and_returns_ppo_metrics():
    # Arrange
    learner, _ = _learner(_config())

    # Act
    result = learner.run_iteration(0)

    # Assert
    assert isinstance(result, IterationResult)
    assert set(result.extra) == EXTRA_KEYS
    assert all(math.isfinite(value) for value in result.extra.values())
    assert result.extra["std"] == pytest.approx(PPOParams().initial_std, rel=0.2)
    assert math.isfinite(result.best) and math.isfinite(result.mean)
    assert result.params["ppo.learning_rate"] == PPOParams().learning_rate
    assert 0 <= result.finished


def test_episodes_cut_by_the_time_limit_are_counted_and_cars_respawn():
    # Arrange
    learner, fleet = _learner(_config())

    # Act: 4 iterations of 16 steps run past max_steps=50, so every car ends at least one episode
    results = [learner.run_iteration(index) for index in range(4)]

    # Assert
    assert sum(result.extra["episodes"] for result in results) >= 6
    assert fleet.steps_alive.max() <= 50


def test_dead_cars_are_respawned_during_the_rollout():
    # Arrange
    config = _config(
        max_steps=1000, ppo={"rollout_steps": 256, "minibatch_size": 256, "initial_std": 2.0}
    )
    learner, fleet = _learner(config)
    calls: list[tuple[np.ndarray, np.ndarray]] = []
    original = fleet.respawn

    def spy(mask):
        calls.append((mask.copy(), fleet.alive.copy()))
        return original(mask)

    fleet.respawn = spy  # type: ignore[method-assign]

    # Act
    learner.run_iteration(0)

    # Assert
    assert any((~alive).any() for _, alive in calls)
    for mask, alive in calls:
        assert mask[~alive].all()
    assert fleet.alive.all()


def test_same_seed_gives_identical_runs_and_another_seed_differs():
    # Arrange
    first, _ = _learner(_config())
    second, _ = _learner(_config())
    other, _ = _learner(_config(seed=6))

    # Act
    rows = [
        [(r.best, r.mean, r.extra["policy_loss"]) for r in (run.run_iteration(i) for i in range(2))]
        for run in (first, second, other)
    ]

    # Assert
    assert rows[0] == rows[1]
    assert first.best_snapshot()["weights"] == second.best_snapshot()["weights"]
    assert rows[0] != rows[2]


def test_best_snapshot_uses_the_shared_weights_format():
    # Arrange
    config = _config()
    learner, _ = _learner(config)
    assert learner.best_snapshot() is None

    # Act
    learner.run_iteration(0)
    snapshot = learner.best_snapshot()

    # Assert
    assert snapshot is not None
    assert set(snapshot) == {"sizes", "activation", "generation", "fitness", "weights"}
    assert snapshot["sizes"] == config.model.layer_sizes
    assert snapshot["activation"] == config.model.activation
    assert snapshot["generation"] == 0
    assert len(snapshot["weights"]) == config.model.weight_count
    assert math.isfinite(snapshot["fitness"])


def test_hot_parameters_apply_to_the_next_iteration():
    # Arrange
    learner, _ = _learner(_config())
    learner.run_iteration(0)

    # Act
    notices = learner.apply_update(
        {
            "ppo": {"learning_rate": 1e-3, "entropy_coef": 0.02, "crash_penalty": 3.0},
            "fitness": {"weights": {"progress": 1.0}},
        }
    )
    result = learner.run_iteration(1)

    # Assert
    assert notices == []
    assert result.params["ppo.learning_rate"] == 1e-3
    assert result.params["ppo.entropy_coef"] == 0.02
    assert result.params["ppo.crash_penalty"] == 3.0
    assert learner.config.fitness.weights == {"progress": 1.0}


def test_structural_unknown_and_invalid_updates_are_rejected_with_russian_notices():
    # Arrange
    learner, _ = _learner(_config())

    # Act
    notices = learner.apply_update(
        {
            "ppo": {"rollout_steps": 32, "gamma": 2.0, "bogus": 1},
            "population": 10,
            "mutation_rate": 0.5,
        }
    )

    # Assert
    assert len(notices) == 5
    assert "ppo.rollout_steps" in notices[0] and "нельзя менять" in notices[0]
    assert "ppo.gamma" in notices[1] and "не применён" in notices[1]
    assert "ppo.bogus" in notices[2] and "неизвестный" in notices[2]
    assert "population" in notices[3] and "нельзя менять" in notices[3]
    assert "mutation_rate" in notices[4]
    assert learner.config.ppo.rollout_steps == 16
    assert learner.config.ppo.gamma == PPOParams().gamma


def test_ppo_update_must_be_a_mapping():
    # Arrange
    learner, _ = _learner(_config())

    # Act
    notices = learner.apply_update({"ppo": 5})

    # Assert
    assert len(notices) == 1 and "ppo" in notices[0]


def test_frames_reach_the_view_with_network_state_every_step():
    # Arrange
    class View:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def draw(self, fleet, lines, network) -> None:
            self.calls.append(network)

    view = View()
    learner, _ = _learner(_config(), view)

    # Act
    learner.run_iteration(0)

    # Assert
    assert len(view.calls) == 16
    matrices, activations = view.calls[0]
    assert len(matrices) == len(ModelSpec().layer_sizes) - 1
    assert len(activations) == len(ModelSpec().layer_sizes)


def test_schema_is_the_ppo_schema_and_registry_creates_the_learner():
    # Arrange / Act / Assert
    assert PPOLearner.schema() == PPO_SCHEMA
    assert isinstance(get_learner("ppo"), PPOLearner)


def test_importing_the_lab_and_ppo_modules_does_not_load_torch_until_a_learner_exists():
    # Arrange
    code = (
        "import sys\n"
        "import rl_fun.lab.worker, rl_fun.lab.catalog, rl_fun.learners.registry\n"
        "import rl_fun.learners.ppo as ppo\n"
        "assert 'torch' not in sys.modules, 'torch loaded at import time'\n"
        "ppo.PPOLearner()\n"
        "assert 'torch' in sys.modules, 'torch not loaded by the learner'\n"
        "print('ok')\n"
    )

    # Act
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )

    # Assert
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
