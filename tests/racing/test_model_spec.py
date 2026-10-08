import json

import pytest

from rl_fun.racing.model_spec import ModelSpec, available_inputs, available_outputs

DEFAULT_INPUTS = (
    "ray:-90", "ray:-30", "ray:0", "ray:30", "ray:90", "speed", "lateral_speed", "yaw_rate",
)


def test_default_spec_matches_current_model():
    spec = ModelSpec()

    assert spec.inputs == DEFAULT_INPUTS
    assert spec.hidden == (6, 5)
    assert spec.outputs == ("steer", "throttle")
    assert spec.activation == "tanh"
    assert spec.layer_sizes == [8, 6, 5, 2]
    assert spec.weight_count == (8 + 1) * 6 + (6 + 1) * 5 + (5 + 1) * 2


def test_ray_angles_follow_input_order():
    spec = ModelSpec(inputs=("speed", "ray:30", "ray:-30", "yaw_rate"))

    assert spec.ray_angles_deg == (30.0, -30.0)


def test_labels_for_default_spec():
    spec = ModelSpec()

    assert spec.input_labels == [
        "↑ -90°", "↑ -30°", "↑ 0°", "↑ 30°", "↑ 90°", "Скор.", "Бок.", "Угл.",
    ]
    assert spec.output_labels == ["Руль", "Газ"]


def test_labels_for_extended_spec():
    spec = ModelSpec(
        inputs=("ray:0", "acceleration", "steering_angle"),
        outputs=("steer", "accelerate", "brake", "boost"),
    )

    assert spec.input_labels == ["↑ 0°", "Уск.", "Колёса"]
    assert spec.output_labels == ["Руль", "Газ", "Тормоз", "Буст"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inputs": ()},
        {"inputs": ("speed", "speed")},
        {"inputs": ("warp",)},
        {"inputs": ("ray:abc",)},
        {"inputs": ("ray:200",)},
        {"hidden": (0,)},
        {"hidden": (-3,)},
        {"outputs": ("steer",)},
        {"outputs": ("throttle",)},
        {"outputs": ("steer", "throttle", "accelerate")},
        {"outputs": ("steer", "throttle", "brake")},
        {"outputs": ("steer", "warp")},
        {"outputs": ("steer", "throttle", "throttle")},
        {"activation": "swish"},
    ],
)
def test_invalid_specs_raise(kwargs: dict):
    with pytest.raises(ValueError):
        ModelSpec(**kwargs)


def test_empty_hidden_layers_are_allowed():
    spec = ModelSpec(hidden=())

    assert spec.layer_sizes == [8, 2]


def test_dict_round_trip_through_json():
    spec = ModelSpec(
        inputs=("ray:-45", "ray:45", "speed", "acceleration"),
        hidden=(4,),
        outputs=("steer", "accelerate", "brake", "boost"),
        activation="relu",
    )

    restored = ModelSpec.from_dict(json.loads(json.dumps(spec.to_dict())))

    assert restored == spec


def test_from_dict_uses_defaults_and_rejects_unknown_keys():
    assert ModelSpec.from_dict({}) == ModelSpec()
    with pytest.raises(ValueError, match="bogus"):
        ModelSpec.from_dict({"bogus": 1})


def test_catalogs_list_choices():
    assert {"speed", "lateral_speed", "yaw_rate", "acceleration", "steering_angle"} <= set(
        available_inputs()
    )
    assert available_outputs() == ("steer", "throttle", "accelerate", "brake", "boost")
