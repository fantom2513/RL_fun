"""Learner registry: maps a learner name to a factory. Heavy dependencies load only on demand."""

from __future__ import annotations

from collections.abc import Callable

from rl_fun.learners.base import Learner


def _evolution() -> Learner:
    from rl_fun.learners.evolution import EvolutionLearner

    return EvolutionLearner()


def _ppo() -> Learner:
    from rl_fun.learners.ppo import PPOLearner

    return PPOLearner()


def _cem() -> Learner:
    from rl_fun.learners.cem import CEMLearner

    return CEMLearner()


def _a2c() -> Learner:
    from rl_fun.learners.a2c import A2CLearner

    return A2CLearner()


_FACTORIES: dict[str, Callable[[], Learner]] = {
    "evolution": _evolution,
    "ppo": _ppo,
    "cem": _cem,
    "a2c": _a2c,
}


def available_learners() -> tuple[str, ...]:
    """Names of the registered learners, in catalog order."""
    return tuple(_FACTORIES)


def get_learner(name: str) -> Learner:
    """A fresh learner by name; an unknown or not yet implemented name raises `ValueError`."""
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise ValueError(
            f"неизвестный обучатель {name!r}; допустимые: {', '.join(_FACTORIES)}"
        ) from None
    return factory()
