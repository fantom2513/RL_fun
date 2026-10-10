"""Atomic tensor-only checkpoints; no arbitrary-object deserialization."""

import os
import uuid
from pathlib import Path

import torch

from rl_fun.snake.config import SnakeConfig
from rl_fun.snake.policy import ARCHITECTURE


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("неподдерживаемая версия модели")
        if state.get("architecture") != ARCHITECTURE:
            raise ValueError("несовместимая архитектура модели")
        config = SnakeConfig.from_dict(state["config"])
        if state.get("algorithm") != config.algorithm:
            raise ValueError("несовместимый алгоритм модели")
        from rl_fun.snake.policy import Policy

        model = Policy()
        model.load_state_dict(state["model"])
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("модель содержит некорректные веса")
        return state
    except Exception as error:
        raise ValueError("не удалось загрузить совместимую модель змейки") from error
