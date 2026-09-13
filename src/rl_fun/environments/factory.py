"""Factory for Gymnasium environments used by the runtime."""

import gymnasium as gym

from rl_fun.environments import register_environments
from rl_fun.experiments.config import EnvironmentConfig


def make_environment(config: EnvironmentConfig, render_mode: str | None = None) -> gym.Env:
    """Create an environment from config with an optional Gymnasium render mode."""
    register_environments()
    kwargs = dict(config.kwargs)
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    try:
        return gym.make(config.id, **kwargs)
    except Exception as error:
        raise ValueError(
            f"unable to create environment {config.id!r} with kwargs {sorted(kwargs)}"
        ) from error
