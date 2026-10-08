"""Interactive keyboard driving for the racing environment."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import pygame

from rl_fun.environments import register_environments


def action_from_keys(left: bool, right: bool, up: bool, down: bool) -> np.ndarray:
    """Map pressed keys to a [steer, throttle] action; positive steer turns left."""
    return np.array([float(left) - float(right), float(up) - float(down)], dtype=np.float32)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ручное вождение по гоночной трассе. Стрелки или WASD — управление, "
        "R — перезапуск, Esc — выход. Колесо мыши или +/- — масштаб, C — переключить камеру "
        "(вся трасса / слежение за машиной). Размер окна можно менять мышью.",
    )
    parser.add_argument(
        "--camera",
        choices=("fit", "follow"),
        default="fit",
        help="Камера: fit — вся трасса, follow — слежение за машиной (по умолчанию fit)",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=2.0,
        help="Масштаб камеры слежения от 0.5 до 6 (по умолчанию 2)",
    )
    parser.add_argument(
        "track", nargs="?", default="oval", help="Имя встроенной трассы или путь к JSON-файлу"
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Лимит шагов эпизода")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Остановиться после N кадров (для проверок)"
    )
    return parser.parse_args(argv)


def _summary(info: dict) -> str:
    if info["collided"]:
        outcome = "авария"
    elif info["lap"] >= 1:
        outcome = "круг пройден"
    else:
        outcome = "время вышло"
    return f"Эпизод завершён: {outcome}, прогресс {info['progress']:.0%}"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the interactive game loop."""
    parser = argparse.ArgumentParser()  # only used to report errors in the same style
    arguments = parse_args(argv)
    register_environments()
    options = {"track": arguments.track, "render_mode": "human"}
    if arguments.max_steps is not None:
        options["max_steps"] = arguments.max_steps
    try:
        env = gym.make("RLFun/Racing-v0", **options)
    except (OSError, ValueError) as error:
        parser.error(f"Не удалось создать среду: {error}")

    env.unwrapped.render_camera = arguments.camera
    env.unwrapped.render_zoom = arguments.zoom
    frames = 0
    try:
        env.reset()
        while arguments.max_frames is None or frames < arguments.max_frames:
            for event in pygame.event.get():
                env.unwrapped.handle_render_event(event)
                if event.type == pygame.QUIT:
                    return 0
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return 0
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    env.reset()
            keys = pygame.key.get_pressed()
            action = action_from_keys(
                left=keys[pygame.K_LEFT] or keys[pygame.K_a],
                right=keys[pygame.K_RIGHT] or keys[pygame.K_d],
                up=keys[pygame.K_UP] or keys[pygame.K_w],
                down=keys[pygame.K_DOWN] or keys[pygame.K_s],
            )
            _, _, terminated, truncated, info = env.step(action)
            frames += 1
            if terminated or truncated:
                print(_summary(info))
                env.reset()
        return 0
    finally:
        env.close()
