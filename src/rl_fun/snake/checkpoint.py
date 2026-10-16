"""Atomic tensor-only checkpoints; no arbitrary-object deserialization."""

import math
import os
import uuid
from itertools import chain
from pathlib import Path

import torch

from rl_fun.snake.config import SnakeConfig
from rl_fun.snake.policy import ARCHITECTURE, architecture


def playback_weights(state):
    if state.get("best_rank", [-1])[0] >= 0:
        return state["best_model"]
    return state["model"]


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_optimizer(saved, parameters):
    """Check Adam moments before accepting a resumable checkpoint."""
    parameters = list(parameters)
    optimizer = torch.optim.Adam(parameters)
    optimizer.load_state_dict(saved)

    def finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(item) for item in value)
        return True

    if not finite(saved):
        raise ValueError("оптимизатор содержит некорректные значения")
    for parameter, moments in optimizer.state.items():
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in moments and moments[key].shape != parameter.shape:
                raise ValueError("несовместимая форма состояния оптимизатора")


def load_checkpoint(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or state.get("version") not in (1, 2):
            raise ValueError("неподдерживаемая версия модели")
        config = SnakeConfig.from_dict(state["config"])
        expected = (
            ARCHITECTURE
            if state["version"] == 1
            else architecture(config.size, config.observation_mode)
        )
        if state.get("architecture") != expected:
            raise ValueError("несовместимая архитектура модели")
        if state["version"] == 1 and config.observation_mode != "local":
            raise ValueError("несовместимое наблюдение старой модели")
        if state.get("algorithm") != config.algorithm:
            raise ValueError("несовместимый алгоритм модели")
        from rl_fun.snake.policy import Policy

        model = Policy(config.size, config.observation_mode)
        model.load_state_dict(state["model"])
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("модель содержит некорректные веса")
        if config.algorithm in ("dqn", "double_dqn", "td3"):
            validate_optimizer(state["optimizer"], model.parameters())
            model.load_state_dict(state["target"])
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("целевая модель содержит некорректные веса")
            replay = state["replay"]
            count = len(replay["actions"])
            if count > config.replay_capacity:
                raise ValueError("память опыта превышает вместимость")
            for key in ("observations", "next"):
                value = replay[key]
                if count == 0 and value.numel() == 0:
                    continue  # legacy empty replay has shape (0,)
                if value.shape != (count, expected[0]) or not torch.isfinite(value).all():
                    raise ValueError("несовместимое наблюдение в памяти опыта")
            for key in ("actions", "rewards", "done"):
                shape = (count, 3) if key == "actions" and config.algorithm == "td3" else (count,)
                if replay[key].shape != shape or not torch.isfinite(replay[key]).all():
                    raise ValueError("несовместимые переходы в памяти опыта")
            action_type = torch.float32 if config.algorithm == "td3" else torch.long
            if replay["actions"].dtype != action_type or replay["done"].dtype != torch.bool:
                raise ValueError("некорректный тип переходов в памяти опыта")
            valid = (
                (replay["actions"].abs() <= 1)
                if config.algorithm == "td3"
                else ((replay["actions"] >= 0) & (replay["actions"] < 3))
            )
            if count and not valid.all():
                raise ValueError("некорректные действия в памяти опыта")
            if config.algorithm == "td3":
                from rl_fun.snake.td3 import make_q_critic

                if state.get("action_adapter") != "continuous_scores_argmax":
                    raise ValueError("несовместимый адаптер действий TD3")
                critic = make_q_critic(config)
                for key in ("critic1", "critic2", "target_critic1", "target_critic2"):
                    critic.load_state_dict(state[key])
                    if any(not torch.isfinite(p).all() for p in critic.parameters()):
                        raise ValueError("критик содержит некорректные веса")
                validate_optimizer(
                    state["critic_optimizer"],
                    chain(critic.parameters(), make_q_critic(config).parameters()),
                )
        elif config.algorithm in ("reinforce", "a2c", "ppo"):
            from rl_fun.snake.on_policy import ROLLOUT_KEYS, make_critic

            critic = make_critic(config)
            critic.load_state_dict(state["critic"])
            if any(not torch.isfinite(p).all() for p in critic.parameters()):
                raise ValueError("оценщик содержит некорректные веса")
            validate_optimizer(state["optimizer"], chain(model.parameters(), critic.parameters()))
            rollout = state["rollout"]
            count = len(rollout["actions"])
            limit = config.max_steps if config.algorithm == "reinforce" else config.rollout_steps
            if count > limit or rollout["observations"].shape != (count, expected[0]):
                raise ValueError("несовместимая траектория обучения")
            for key in ROLLOUT_KEYS:
                if not torch.isfinite(rollout[key]).all():
                    raise ValueError("траектория содержит некорректные значения")
                if key != "observations" and rollout[key].shape != (count,):
                    raise ValueError("несовместимая траектория обучения")
            if (
                rollout["actions"].dtype != torch.long
                or any(rollout[key].dtype != torch.bool for key in ("terminated", "ended"))
                or not ((rollout["actions"] >= 0) & (rollout["actions"] < 3)).all()
            ):
                raise ValueError("некорректные переходы в траектории")
            cursor = state.get("optimization")
            if cursor is not None:
                if (
                    not count
                    or not 0 <= cursor["epoch"] < cursor["epochs"] <= 50
                    or not 0 <= cursor["offset"] <= count
                    or not 1 <= cursor["batch_size"] <= max(count, 512)
                ):
                    raise ValueError("несовместимый курсор обновления политики")
                indices = cursor["indices"]
                if indices is not None and (
                    indices.dtype != torch.long
                    or indices.shape != (count,)
                    or not torch.equal(indices.sort().values, torch.arange(count))
                ):
                    raise ValueError("несовместимый порядок обновления политики")
        if "best_model" in state:
            model.load_state_dict(state["best_model"])
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("лучшая модель содержит некорректные веса")
        return state
    except Exception as error:
        raise ValueError("не удалось загрузить совместимую модель змейки") from error
