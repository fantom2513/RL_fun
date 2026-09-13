"""Gymnasium environments used by the RL workbench."""

import gymnasium as gym

LOCAL_ENVIRONMENTS = {
    "RLFun/StationaryBandit-v0": "rl_fun.environments.bandit:StationaryBanditEnv",
}


def register_environments() -> None:
    """Register project environments without replacing another registration."""
    for environment_id, entry_point in LOCAL_ENVIRONMENTS.items():
        existing = gym.registry.get(environment_id)
        if existing is None:
            gym.register(id=environment_id, entry_point=entry_point)
        elif existing.entry_point != entry_point:
            raise RuntimeError(
                f"environment {environment_id!r} is already registered "
                f"with entry point {existing.entry_point!r}"
            )


register_environments()
