import argparse
import json
from pathlib import Path

from rl_fun.experiments.compare import aggregate_batch
from rl_fun.experiments.config import ExperimentConfig
from rl_fun.experiments.runner import run_many


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an RL Fun experiment")
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    try:
        config = ExperimentConfig.from_json(arguments.config)
        batch = run_many(config)
    except KeyboardInterrupt:
        return 130
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(aggregate_batch(batch), indent=2, sort_keys=True))
    return 0 if batch.failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
