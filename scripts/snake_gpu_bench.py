"""Measure batched Snake throughput: environment steps per second, observations included."""

import argparse
import time

import torch

from rl_fun.snake.gpu.vec_env import VecSnake


def measure(size, num_envs, device, steps, warmup=20, observe=True):
    vec = VecSnake(size, num_envs, device, seed=0)
    generator = torch.Generator(device=device).manual_seed(0)

    def run(count):
        for _ in range(count):
            vec.step(torch.randint(3, (num_envs,), generator=generator, device=device))
            if observe:
                vec.observation()

    run(warmup)
    if vec.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    run(steps)
    if vec.device.type == "cuda":
        torch.cuda.synchronize()
    return steps * num_envs / (time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-observation", action="store_true")
    args = parser.parse_args()
    rate = measure(args.size, args.envs, args.device, args.steps, observe=not args.no_observation)
    print(
        f"{args.device} size={args.size} envs={args.envs} "
        f"observation={not args.no_observation}: {rate:,.0f} steps/s"
    )


if __name__ == "__main__":
    main()
