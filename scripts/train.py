import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rl_fun.experiments.compare import aggregate_batch
from rl_fun.experiments.config import ExperimentConfig
from rl_fun.experiments.runner import run_many


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Запустить эксперимент RL Fun для всех seed")
    parser.add_argument("config", type=Path, help="Путь к конфигурации версии 2")
    parser.add_argument(
        "--metric", default="train/cumulative_reward", help="Ключ метрики для сводной статистики",
    )
    arguments = parser.parse_args(argv)
    try:
        config = ExperimentConfig.from_json(arguments.config)
        batch = run_many(config)
        aggregate = aggregate_batch(batch, arguments.metric)
    except KeyboardInterrupt:
        return 130
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(f"Не удалось выполнить эксперимент: {error}")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0 if batch.failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
