import argparse
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import numpy as np

from rl_fun.environments.factory import make_environment
from rl_fun.experiments.config import ExperimentConfig
from rl_fun.policies import RandomPolicy
from rl_fun.rollouts import rollout_episode
from rl_fun.tracking.artifacts import collect_metadata, create_run_dir, write_json_atomic
from rl_fun.tracking.metrics import JsonlMetricSink


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Оценить случайную policy без открытия окна")
    parser.add_argument("config", type=Path, help="Путь к конфигурации версии 2")
    parser.add_argument(
        "--seed", type=int, help="Seed оценки (по умолчанию первый из конфигурации)",
    )
    arguments = parser.parse_args(argv)
    try:
        config = ExperimentConfig.from_json(arguments.config)
        seed = config.run.seeds[0] if arguments.seed is None else arguments.seed
        if seed < 0:
            raise ValueError("seed должен быть неотрицательным")
        rng = np.random.default_rng(seed)
        run_dir = create_run_dir(
            Path(config.run.output_root), config.name, f"eval-seed-{seed}-{uuid4().hex[:12]}",
        )
        write_json_atomic(run_dir / "config.json", config.to_dict())
        write_json_atomic(run_dir / "metadata.json", collect_metadata(seed))
        rewards = []
        env = make_environment(config.environment)
        try:
            policy = RandomPolicy()
            with JsonlMetricSink(run_dir / "metrics.jsonl") as metrics:
                for episode_index in range(config.evaluation.episodes):
                    episode_seed = int(rng.integers(0, np.iinfo(np.uint32).max, endpoint=True))
                    result = rollout_episode(
                        env, policy, episode_seed, config.evaluation.max_episode_steps,
                    )
                    rewards.append(result.reward)
                    metrics.log(
                        episode_index + 1,
                        {"episode/reward": result.reward, "episode/length": float(result.length)},
                    )
        finally:
            env.close()
    except KeyboardInterrupt:
        return 130
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(f"Не удалось выполнить оценку: {error}")
    print(f"Эпизодов: {len(rewards)}")
    print(f"Артефакты: {run_dir}")
    print(f"Средняя награда: {float(np.mean(rewards))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
