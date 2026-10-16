"""Compare bounded Snake experiments without using held-out validation seeds."""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rl_fun.snake.benchmark import run_benchmark, write_json
from rl_fun.snake.config import SnakeConfig

ALGORITHMS = ("dqn", "double_dqn", "reinforce", "a2c", "ppo", "td3")


def experiment(spec, runs_dir, steps, seconds):
    config = SnakeConfig.from_dict(spec["config"])
    directory = run_benchmark(
        config,
        runs_dir,
        seconds,
        spec.get("resume"),
        train_steps=steps,
        progress=lambda message: print(f"[{spec['name']}] {message}", flush=True),
    )
    report = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    return {
        "name": spec["name"],
        "directory": str(directory),
        "config": config.to_dict(),
        **{
            key: report[key]
            for key in (
                "wins",
                "episodes",
                "win_rate",
                "occupancy",
                "max_occupancy",
                "score",
                "reasons",
                "completed",
                "steps",
                "steps_added",
                "updates",
                "pending_rollout_steps",
                "training_seconds",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs/lab")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--configs", help="JSON list of named config/resume experiments")
    parser.add_argument(
        "--manifest", help="JSON result path; updated after each finished experiment"
    )
    args = parser.parse_args()
    if args.steps < 1 or not 0 < args.seconds < float("inf"):
        parser.error("steps and seconds must be finite and positive")
    if args.configs:
        specs = json.loads(Path(args.configs).read_text(encoding="utf-8"))
    else:
        specs = [
            {
                "name": algorithm,
                "config": SnakeConfig(
                    algorithm=algorithm,
                    observation_mode="board",
                    size=16,
                    seed=42,
                    episodes=100000,
                    max_steps=100000,
                    learning_rate=0.0003,
                    gamma=0.99,
                    batch_size=128,
                    replay_capacity=50000,
                    warmup=1000,
                    epsilon_decay=1000,
                    epsilon_end=0.02,
                    target_interval=500,
                    food_reward=1,
                    death_penalty=1,
                    win_reward=10,
                    step_penalty=0,
                    entropy_coefficient=0.02,
                    rollout_steps=512,
                    evaluation_interval=1000,
                    evaluation_episodes=20,
                ).to_dict(),
            }
            for algorithm in args.algorithms
        ]
    if not specs or len({spec["name"] for spec in specs}) != len(specs):
        parser.error("experiments need unique names")
    for spec in specs:
        config = SnakeConfig.from_dict(spec["config"])
        if (
            config.size < 16
            or config.observation_mode not in ("board", "egocentric")
            or config.algorithm not in ALGORITHMS
        ):
            parser.error(
                "completion search requires board observations, size>=16 and a supported algorithm"
            )
    path = Path(args.manifest or f"{args.runs_dir}/snake/search-{time.time_ns()}.json").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "phase": "tuning",
        "status": "running",
        "steps_budget_per_run": args.steps,
        "seconds_cap_per_run": args.seconds,
        "experiments": specs,
        "results": [],
        "errors": [],
    }
    write_json(path, manifest)
    print(f"Manifest: {path}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs = {
            executor.submit(experiment, spec, args.runs_dir, args.steps, args.seconds): spec
            for spec in specs
        }
        for future in as_completed(jobs):
            spec = jobs[future]
            try:
                result = future.result()
                manifest["results"].append(result)
                manifest["results"].sort(
                    key=lambda row: (row["win_rate"], row["occupancy"]), reverse=True
                )
                print(
                    f"Finished {spec['name']}: {result['wins']}/{result['episodes']} wins, "
                    f"{result['occupancy']:.1%} occupancy, {result['steps_added']} steps",
                    flush=True,
                )
            except Exception as error:
                manifest["errors"].append({"name": spec["name"], "error": str(error)})
            write_json(path, manifest)
    manifest["status"] = "finished" if not manifest["errors"] else "error"
    write_json(path, manifest)
    if manifest["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
