import numpy as np
import pytest

from rl_fun.racing import reference as ref
from rl_fun.racing.fitness import FitnessSpec
from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.fleet_view import FleetView
from rl_fun.racing.model_spec import ModelSpec


def test_relu_and_sigmoid_hidden_layers_change_the_output():
    sizes = [4, 5, 2]
    weights = ref.init_population(np.random.default_rng(1), 1, sizes, 1.0)[0]
    observation = np.array([0.3, -0.7, 0.9, 0.1])

    tanh = ref.forward(weights, observation, sizes)
    relu = ref.forward(weights, observation, sizes, activation="relu")
    sigmoid = ref.forward(weights, observation, sizes, activation="sigmoid")

    assert not np.allclose(tanh, relu)
    assert not np.allclose(tanh, sigmoid)
    for action in (tanh, relu, sigmoid):
        assert np.all(np.abs(action) <= 1.0)


def test_inspect_uses_the_same_activation():
    sizes = [4, 5, 2]
    weights = ref.init_population(np.random.default_rng(1), 1, sizes, 1.0)[0]
    observation = np.array([0.3, -0.7, 0.9, 0.1])

    _, activations = ref.inspect(weights, observation, sizes, activation="relu")

    assert activations[-1] == pytest.approx(
        ref.forward(weights, observation, sizes, activation="relu")
    )
    assert np.all(activations[1] >= 0.0)


def test_unknown_activation_raises():
    with pytest.raises(ValueError):
        ref.forward(np.zeros(10), np.zeros(2), [2, 2], activation="swish")


def test_train_accepts_a_custom_model_and_fitness():
    model = ModelSpec(
        inputs=("ray:-45", "ray:0", "ray:45", "speed"),
        hidden=(4,),
        outputs=("steer", "accelerate", "brake"),
    )
    params = ref.EvolutionParams(population=12, elite=3)

    run = ref.train(
        "oval", seed=3, generations=3, params=params, fleet_kwargs={"max_steps": 150},
        model=model, fitness=FitnessSpec({"progress": 1.0, "centering": 0.2}),
    )

    assert run.sizes == model.layer_sizes
    assert run.best_weights.shape == (model.weight_count,)
    assert len(run.history) == 3


def test_default_training_is_unchanged():
    params = ref.EvolutionParams(population=12, elite=3)

    run = ref.train("oval", seed=3, generations=2, params=params, fleet_kwargs={"max_steps": 150})

    assert run.sizes == ref.layer_sizes(8, params.hidden)


def test_fleet_view_takes_labels_from_the_fleet_model():
    model = ModelSpec(
        inputs=("ray:0", "speed"), hidden=(3,), outputs=("steer", "accelerate", "brake")
    )
    fleet = RacingFleet(3, model=model)
    fleet.reset()
    view = FleetView(fleet, mode="rgb_array", show_network=True)
    try:
        assert view.panel is not None
        assert list(view.panel.input_labels) == model.input_labels
        assert list(view.panel.output_labels) == model.output_labels
    finally:
        view.close()
