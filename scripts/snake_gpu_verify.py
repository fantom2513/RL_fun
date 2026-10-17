"""Acceptance run: greedy, unmasked play of a GPU Snake model on the reference ``SnakeEnv``."""

import argparse
import json
import time
from pathlib import Path

import torch

from rl_fun.snake.gpu.checkpoint import load_model
from rl_fun.snake.gpu.evaluate import evaluate_reference_batched
from rl_fun.snake.presets import benchmark_seeds, verification_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--tuning", action="store_true", help="use the 20 tuning seeds instead")
    args = parser.parse_args()

    tuning, validation = benchmark_seeds()
    seeds = tuning if args.tuning else validation
    model, meta = load_model(args.checkpoint, size=args.size)
    started = time.perf_counter()
    report = evaluate_reference_batched(
        model, args.size, seeds, args.max_steps or args.size**4, device=args.device
    )
    report["seconds"] = round(time.perf_counter() - started, 1)
    report["checkpoint"] = str(args.checkpoint)
    report["checkpoint_meta"] = meta
    report["seeds"] = [seeds[0], seeds[-1]]
    report["backend"] = "reference SnakeEnv (egocentric_cycle), greedy, no mask, no teacher"
    report["status"] = verification_status(args.size, report["games"], report["wins"])
    target = Path(args.checkpoint).with_name(
        "verify-tuning.json" if args.tuning else "verify-validation.json"
    )
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k not in ("outcomes", "checkpoint_meta")}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
