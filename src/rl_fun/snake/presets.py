"""Explicit candidate settings and evidence thresholds for completion presets."""

from rl_fun.snake.config import SnakeConfig


def benchmark_seeds():
    return list(range(1000000, 1000020)), list(range(3000000, 3000100))


def verification_status(size, episodes, wins):
    return (
        "verified" if size >= 16 and episodes >= 100 and wins / episodes >= 0.9 else "experimental"
    )


def completion_preset():
    config = SnakeConfig(
        algorithm="double_dqn",
        observation_mode="board",
        size=16,
        episodes=100000,
        max_steps=100000,
        learning_rate=0.0003,
        gamma=0.999,
        replay_capacity=50000,
        warmup=1000,
        train_interval=4,
        target_interval=500,
        epsilon_decay=60000,
        epsilon_end=0.02,
        win_reward=100,
        step_penalty=0.001,
        evaluation_episodes=20,
        evaluation_interval=200,
    )
    return {
        "id": "completion_16",
        "label": "Полное поле 16×16 · экспериментальный",
        "status": "experimental",
        "config": config.to_dict(),
        "visualization": {"enabled": False, "fps": 60},
        "description": (
            "Double DQN с картой тела. Проверенный результат — от 90 побед из 100; "
            "параметры пока требуют обучения."
        ),
    }
