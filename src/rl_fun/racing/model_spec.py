"""Data description of a racing car's network: inputs, hidden layers, outputs, activation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

_RAY_PREFIX = "ray:"
_SCALAR_INPUTS = ("speed", "lateral_speed", "yaw_rate", "acceleration", "steering_angle")
_OUTPUTS = ("steer", "throttle", "accelerate", "brake", "boost")
_ACTIVATIONS = ("tanh", "relu", "sigmoid")
_INPUT_LABELS = {
    "speed": "Скор.",
    "lateral_speed": "Бок.",
    "yaw_rate": "Угл.",
    "acceleration": "Уск.",
    "steering_angle": "Колёса",
}
_OUTPUT_LABELS = {
    "steer": "Руль",
    "throttle": "Газ",
    "accelerate": "Газ",
    "brake": "Тормоз",
    "boost": "Буст",
}


def available_inputs() -> tuple[str, ...]:
    """Input names other than rays, which take the form ``ray:<angle in degrees>``."""
    return _SCALAR_INPUTS


def available_outputs() -> tuple[str, ...]:
    """Output channel names understood by the controls mapping."""
    return _OUTPUTS


def _parse_ray(name: str) -> float:
    try:
        angle = float(name[len(_RAY_PREFIX) :])
    except ValueError as error:
        raise ValueError(f"invalid ray input {name!r}: angle must be a number") from error
    if not -180.0 <= angle <= 180.0:
        raise ValueError(f"invalid ray input {name!r}: angle must be within -180..180")
    return angle


def _validate_inputs(inputs: tuple[str, ...]) -> None:
    if not inputs:
        raise ValueError("inputs must not be empty")
    if len(set(inputs)) != len(inputs):
        raise ValueError(f"inputs must be unique: {inputs}")
    for name in inputs:
        if name.startswith(_RAY_PREFIX):
            _parse_ray(name)
        elif name not in _SCALAR_INPUTS:
            raise ValueError(f"unknown input {name!r}; supported: {_SCALAR_INPUTS}")


def _validate_outputs(outputs: tuple[str, ...]) -> None:
    if len(set(outputs)) != len(outputs):
        raise ValueError(f"outputs must be unique: {outputs}")
    for name in outputs:
        if name not in _OUTPUTS:
            raise ValueError(f"unknown output {name!r}; supported: {_OUTPUTS}")
    if "steer" not in outputs:
        raise ValueError("outputs must include 'steer'")
    if "throttle" in outputs:
        if "accelerate" in outputs or "brake" in outputs:
            raise ValueError("'throttle' cannot be combined with 'accelerate' or 'brake'")
    elif "accelerate" not in outputs:
        raise ValueError("outputs need a throttle source: 'throttle' or 'accelerate'")


@dataclass(frozen=True)
class ModelSpec:
    """A car's network described as data. Validated on creation."""

    inputs: tuple[str, ...] = (
        "ray:-90", "ray:-30", "ray:0", "ray:30", "ray:90", "speed", "lateral_speed", "yaw_rate",
    )
    hidden: tuple[int, ...] = (6, 5)
    outputs: tuple[str, ...] = ("steer", "throttle")
    activation: str = "tanh"

    def __post_init__(self) -> None:
        for name in ("inputs", "hidden", "outputs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        _validate_inputs(self.inputs)
        if any(size <= 0 for size in self.hidden):
            raise ValueError(f"hidden layer sizes must be positive: {self.hidden}")
        _validate_outputs(self.outputs)
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {self.activation!r}; supported: {_ACTIVATIONS}")

    @property
    def layer_sizes(self) -> list[int]:
        """Neuron counts from input to output layer."""
        return [len(self.inputs), *self.hidden, len(self.outputs)]

    @property
    def weight_count(self) -> int:
        """Number of trainable parameters, biases included."""
        sizes = self.layer_sizes
        return sum((n_in + 1) * n_out for n_in, n_out in zip(sizes[:-1], sizes[1:], strict=True))

    @property
    def ray_angles_deg(self) -> tuple[float, ...]:
        """Ray angles in degrees, in input order."""
        return tuple(_parse_ray(name) for name in self.inputs if name.startswith(_RAY_PREFIX))

    @property
    def input_labels(self) -> list[str]:
        """Short human-readable label for each input."""
        labels = []
        for name in self.inputs:
            if name.startswith(_RAY_PREFIX):
                labels.append(f"↑ {_parse_ray(name):g}°")
            else:
                labels.append(_INPUT_LABELS[name])
        return labels

    @property
    def output_labels(self) -> list[str]:
        """Short human-readable label for each output."""
        return [_OUTPUT_LABELS[name] for name in self.outputs]

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible representation."""
        return {
            "inputs": list(self.inputs),
            "hidden": list(self.hidden),
            "outputs": list(self.outputs),
            "activation": self.activation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelSpec:
        """Build from `to_dict` output. Missing keys take defaults; unknown keys are rejected."""
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown model spec keys: {', '.join(unknown)}")
        return cls(**data)
