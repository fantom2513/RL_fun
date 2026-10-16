"""Train the GPU Snake model: ``clone`` copies the cycle teacher, ``finetune`` runs PPO."""

import argparse
import json
import time
from pathlib import Path

import torch

from rl_fun.snake.gpu.checkpoint import load_model, save_model
from rl_fun.snake.gpu.evaluate import evaluate_reference, evaluate_vec, greedy_policy
from rl_fun.snake.gpu.imitation import clone
from rl_fun.snake.gpu.model import ConvActorCritic
from rl_fun.snake.gpu.ppo import PPOConfig, collect_rollout, finish_rollout, ppo_update
from rl_fun.snake.gpu.vec_env import VecSnake
from rl_fun.snake.presets import benchmark_seeds


def log(handle, record):
    line = json.dumps(record, ensure_ascii=False)
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def run_clone(args):
    if not torch.cuda.is_available() and args.device == "cuda":
        raise SystemExit("CUDA недоступна: установите torch с CUDA или укажите --device cpu")
    device = args.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        model, _ = load_model(args.resume, size=args.size, device=device)
    else:
        model = ConvActorCritic(args.size).to(device)
    vec = VecSnake(
        args.size,
        args.envs,
        device,
        args.seed,
        start_mode="mixed",
        max_start_length=int(args.max_start_fraction * args.size**2),
    )
    optimizer = torch.optim.Adam(
        [p for n, p in model.named_parameters() if not n.startswith("value_head")],
        lr=args.lr,
    )
    tuning_seeds = benchmark_seeds()[0]
    started = time.perf_counter()
    with (out / "log.jsonl").open("a", encoding="utf-8") as handle:
        log(handle, {"event": "start", "args": vars(args), "parameters": model.parameter_count()})
        done = 0
        while done < args.iterations:
            chunk = min(args.eval_every, args.iterations - done)
            history = clone(
                model, vec, chunk, rollout=args.rollout, explore=args.explore, optimizer=optimizer
            )
            done += chunk
            model.eval()
            report = evaluate_vec(
                greedy_policy(model),
                args.size,
                args.eval_games,
                args.seed + 1,
                device,
                max_steps=args.eval_steps,
            )
            model.train()
            log(
                handle,
                {
                    "event": "progress",
                    "iteration": done,
                    "seconds": round(time.perf_counter() - started, 1),
                    "loss": history[-1]["loss"],
                    "accuracy": sum(row["accuracy"] for row in history[-20:]) / len(history[-20:]),
                    "greedy": {k: report[k] for k in ("win_rate", "occupancy", "reasons")},
                },
            )
            save_model(out / "clone.pt", model, {"stage": "clone", "iterations": done})
        model = model.eval().cpu()
        final = evaluate_reference(model, args.size, tuning_seeds, max_steps=args.size**4)
        log(
            handle,
            {
                "event": "tuning",
                "backend": "reference SnakeEnv, greedy, no mask",
                **{k: final[k] for k in ("games", "wins", "win_rate", "occupancy", "reasons")},
            },
        )


def run_finetune(args):
    if not torch.cuda.is_available() and args.device == "cuda":
        raise SystemExit("CUDA недоступна: установите torch с CUDA или укажите --device cpu")
    if not args.resume:
        raise SystemExit("для дообучения укажите --resume с моделью после клонирования")
    device = args.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, _ = load_model(args.resume, size=args.size, device=device)
    reference = None
    if args.kl_coef:
        reference, _ = load_model(args.resume, size=args.size, device=device)
        reference.eval().requires_grad_(False)
    vec = VecSnake(
        args.size,
        args.envs,
        device,
        args.seed,
        start_mode="mixed" if args.start_fraction else "standard",
        max_start_length=max(3, int(args.start_fraction * args.size**2)),
        food_reward=args.food_reward,
        death_penalty=args.death_penalty,
        step_penalty=args.step_penalty,
        win_reward=args.win_reward,
    )
    config = PPOConfig(
        epochs=args.epochs,
        minibatch=args.minibatch,
        entropy_coef=args.entropy,
        kl_coef=args.kl_coef,
        use_mask=args.mask,
    )
    policy_parameters = [p for n, p in model.named_parameters() if not n.startswith("value_head")]
    optimizer = torch.optim.Adam(
        [{"params": policy_parameters}, {"params": model.value_head.parameters()}], lr=args.lr
    )
    critic_optimizer = torch.optim.Adam(model.value_head.parameters(), lr=args.lr * 10)

    def evaluate():
        model.eval()
        report = evaluate_vec(
            greedy_policy(model),
            args.size,
            args.eval_games,
            args.seed + 1,
            device,
            max_steps=args.eval_steps,
        )
        model.train()
        return report

    started = time.perf_counter()
    with (out / "log.jsonl").open("a", encoding="utf-8") as handle:
        log(handle, {"event": "start", "args": vars(args), "parameters": model.parameter_count()})
        best = evaluate()
        best_rank = (best["win_rate"], best["occupancy"])
        save_model(out / "best.pt", model, {"stage": "finetune", "iteration": 0})
        log(
            handle,
            {"event": "baseline", **{k: best[k] for k in ("win_rate", "occupancy", "reasons")}},
        )
        steps_done = 0
        for iteration in range(1, args.iterations + 1):
            warmup = iteration <= args.critic_warmup
            rollout = finish_rollout(
                collect_rollout(model, vec, args.rollout_steps, args.mask), config
            )
            stats = ppo_update(
                model,
                critic_optimizer if warmup else optimizer,
                rollout,
                config,
                reference=reference,
                critic_only=warmup,
            )
            steps_done += args.rollout_steps * args.envs
            if iteration % args.eval_every and iteration != args.iterations:
                continue
            report = evaluate()
            rank = (report["win_rate"], report["occupancy"])
            if rank > best_rank and not warmup:
                best, best_rank = report, rank
                save_model(out / "best.pt", model, {"stage": "finetune", "iteration": iteration})
            save_model(out / "last.pt", model, {"stage": "finetune", "iteration": iteration})
            log(
                handle,
                {
                    "event": "progress",
                    "iteration": iteration,
                    "env_steps": steps_done,
                    "seconds": round(time.perf_counter() - started, 1),
                    "critic_warmup": warmup,
                    "train": {
                        "games": rollout["finished"],
                        "wins": rollout["wins"],
                        **{k: round(v, 5) for k, v in stats.items()},
                    },
                    "greedy": {k: report[k] for k in ("win_rate", "occupancy", "reasons")},
                    "best_win_rate": best_rank[0],
                },
            )
        best_model, _ = load_model(out / "best.pt", size=args.size)
        final = evaluate_reference(
            best_model.eval(), args.size, benchmark_seeds()[0], max_steps=args.size**4
        )
        log(
            handle,
            {
                "event": "tuning",
                "backend": "reference SnakeEnv, greedy, no mask, best checkpoint",
                **{k: final[k] for k in ("games", "wins", "win_rate", "occupancy", "reasons")},
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("clone", "finetune"))
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--rollout", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=8192)
    parser.add_argument("--entropy", type=float, default=0.001)
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--critic-warmup", type=int, default=20)
    parser.add_argument("--start-fraction", type=float, default=0.0)
    parser.add_argument("--food-reward", type=float, default=1.0)
    parser.add_argument("--death-penalty", type=float, default=1.0)
    parser.add_argument("--step-penalty", type=float, default=0.0005)
    parser.add_argument("--win-reward", type=float, default=2.0)
    parser.add_argument(
        "--mask", action="store_true", help="hide fatal moves while training (hides their penalty)"
    )
    parser.add_argument("--explore", type=float, default=0.1)
    parser.add_argument("--max-start-fraction", type=float, default=0.9)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-games", type=int, default=128)
    parser.add_argument("--eval-steps", type=int, help="step cap of periodic evaluation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--out", default="runs/snake-gpu/clone")
    args = parser.parse_args()
    (run_clone if args.stage == "clone" else run_finetune)(args)


if __name__ == "__main__":
    main()
