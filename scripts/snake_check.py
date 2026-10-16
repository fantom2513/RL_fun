"""Train and verify the Snake completion candidate on independent seeds."""

import argparse
from dataclasses import replace

from rl_fun.snake.benchmark import run_benchmark
from rl_fun.snake.checkpoint import load_checkpoint
from rl_fun.snake.config import SnakeConfig
from rl_fun.snake.presets import completion_preset


def positive_seconds(text):
    import math

    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("seconds must be finite and positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs/lab")
    parser.add_argument("--train-seconds", type=positive_seconds, default=60)
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--algorithm", choices=("dqn", "double_dqn", "reinforce", "a2c", "ppo", "td3")
    )
    parser.add_argument("--resume", help="model.pt checkpoint with latest optimizer/replay")
    parser.add_argument("--epsilon-decay", type=int)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="final evaluation of a frozen --resume model; no training",
    )
    args = parser.parse_args()
    base = load_checkpoint(args.resume)["config"] if args.resume else completion_preset()["config"]
    config = SnakeConfig.from_dict(base)
    if not args.validate_only:
        updates = {"episodes": args.episodes}
        if args.seed is not None:
            updates["seed"] = args.seed
        if args.algorithm is not None:
            updates["algorithm"] = args.algorithm
        if args.epsilon_decay is not None:
            updates["epsilon_decay"] = args.epsilon_decay
        config = replace(config, **updates)
    run_benchmark(
        config, args.runs_dir, args.train_seconds, args.resume, validate=args.validate_only
    )


if __name__ == "__main__":
    main()
