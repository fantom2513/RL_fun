"""Seed-fair evolutionary optimization of the shared neural policy."""

import copy

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from rl_fun.snake.policy import ARCHITECTURE, Policy, evaluate


class EvolutionTrainer:
    def __init__(self, config):
        self.config = config
        torch.set_num_threads(1)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.model = Policy()
        self.candidate = Policy()
        self.population = torch.stack(
            [
                parameters_to_vector(Policy().parameters()).detach().clone()
                for _ in range(config.population)
            ]
        )
        self.completed = 0
        self.champion_reward = -float("inf")
        self.pending = None

    def generation_seeds(self):
        return self.rng.integers(0, 2**31, self.config.evaluation_episodes).tolist()

    def evaluate_candidate(self, vector, seeds, control=lambda: True):
        vector_to_parameters(vector.clone(), self.candidate.parameters())
        return evaluate(self.candidate, self.config.size, seeds, self.config.max_steps, control)

    def reproduce(self, fitness):
        ranking = np.argsort(fitness)[::-1].copy()
        elite_count = max(1, int(self.config.population * self.config.elite_fraction))
        survivors = self.population[torch.tensor(ranking[:elite_count])].clone()
        pool = ranking[: max(2, self.config.population // 2)]
        children = []
        for _ in range(self.config.population - elite_count):
            parent = self.population[int(self.rng.choice(pool))]
            noise = torch.tensor(
                self.rng.normal(0, self.config.mutation_scale, parent.shape), dtype=torch.float32
            )
            children.append(parent + noise)
        self.population = torch.cat((survivors, torch.stack(children)))

    def run(self, control, emit):
        while self.completed < self.config.generations:
            if self.pending is None:
                self.pending = {"seeds": self.generation_seeds(), "results": []}
            seeds = self.pending["seeds"]
            results = self.pending["results"]
            for vector in self.population[len(results) :]:
                result = self.evaluate_candidate(vector, seeds, control)
                if result is None:
                    return
                results.append(result)
            fitness = np.asarray([r["reward"] for r in results])
            best_index = int(fitness.argmax())
            best = self.evaluate_candidate(self.population[best_index], [10001, 10002], control)
            if best is None:
                return
            if best["reward"] > self.champion_reward:
                vector_to_parameters(self.population[best_index].clone(), self.model.parameters())
                self.champion_reward = best["reward"]
            champion = evaluate(
                self.model, self.config.size, [10001, 10002], self.config.max_steps, control
            )
            if champion is None:
                return
            self.completed += 1
            self.reproduce(fitness)
            self.pending = None
            emit(
                {
                    "iteration": self.completed,
                    "reward": float(fitness.max()),
                    "best_reward": float(fitness.max()),
                    "mean_reward": float(fitness.mean()),
                    "score": max(r["score"] for r in results),
                    "mean_score": float(np.mean([r["score"] for r in results])),
                    "evaluation_score": champion["score"],
                    "frame": champion["frame"],
                }
            )

    def state_dict(self):
        return {
            "version": 1,
            "architecture": ARCHITECTURE,
            "algorithm": "evolution",
            "config": self.config.to_dict(),
            "model": self.model.state_dict(),
            "population": self.population,
            "completed": self.completed,
            "champion_reward": self.champion_reward,
            "rng": self.rng.bit_generator.state,
            "pending": copy.deepcopy(self.pending),
            "torch_rng": torch.get_rng_state(),
        }

    def load_state_dict(self, state):
        if state["population"].shape != self.population.shape:
            raise ValueError("для продолжения нужна прежняя численность популяции")
        self.model.load_state_dict(state["model"])
        self.population = state["population"].clone()
        self.completed = state["completed"]
        self.champion_reward = state["champion_reward"]
        self.pending = copy.deepcopy(state.get("pending"))
        self.rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
