"""Tensor-only atomic checkpoints for the GPU Snake models."""

from pathlib import Path

import torch

from rl_fun.snake.checkpoint import save_checkpoint
from rl_fun.snake.gpu.model import ConvActorCritic

FORMAT = "snake-gpu"
VERSION = 1


def save_model(path: str | Path, model: ConvActorCritic, meta: dict) -> None:
    save_checkpoint(
        path,
        {
            "format": FORMAT,
            "version": VERSION,
            "size": model.size,
            "config": {"channels": model.channels, "width": model.width, "hidden": model.hidden},
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "meta": dict(meta),
        },
    )


def load_model(
    path: str | Path, size: int | None = None, device: str | torch.device = "cpu"
) -> tuple[ConvActorCritic, dict]:
    """Load a model saved by ``save_model``; reject anything unexpected."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("не удалось прочитать модель: файл повреждён или небезопасен") from error
    if not isinstance(state, dict) or state.get("format") != FORMAT:
        raise ValueError("это не модель змейки на GPU")
    if state.get("version") != VERSION:
        raise ValueError("неподдерживаемая версия модели")
    if size is not None and state.get("size") != size:
        raise ValueError(f"размер поля модели {state.get('size')} не совпадает с {size}")
    try:
        model = ConvActorCritic(state["size"], **state["config"])
        model.load_state_dict(state["model"])
    except (KeyError, TypeError, RuntimeError) as error:
        raise ValueError("несовместимая архитектура модели") from error
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("модель содержит некорректные веса")
    return model.to(device), dict(state.get("meta", {}))
