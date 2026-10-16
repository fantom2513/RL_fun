"""Seed-fair evolutionary optimization of the shared neural policy."""

import copy

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from rl_fun.snake.policy import Policy, architecture, evaluate, evaluation_rank


class EvolutionTrainer:
    def __init__(self, config):
        self.config = config
        torch.set_num_threads(1)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.model = Policy(config.size, config.observation_mode)
        self.candidate = Policy(config.size, config.observation_mode)
        self.population = torch.stack(
            [
                parameters_to_vector(Policy(config.size, config.observation_mode).parameters())
                .detach()
                .clone()
                for _ in range(config.population)
            ]
        )
        self.completed = 0
        self.champion_reward = -float("inf")
        self.champion_rank = (-1.0, -1.0, -1.0)
        self.pending = None

    def generation_seeds(self):
        return self.rng.integers(0, 1000000, self.config.evaluation_episodes).tolist()

    def evaluate_candidate(self, vector, seeds, control=lambda: True, on_frame=None):
        vector_to_parameters(vector.clone(), self.candidate.parameters())
        return evaluate(
            self.candidate,
            self.config.size,
            seeds,
            self.config.max_steps,
            control,
            on_frame,
            self.config.env_options(),
        )

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

    def run(self, control, emit, on_frame=None):
        while self.completed < self.config.generations:
            if self.pending is None:
                self.pending = {"seeds": self.generation_seeds(), "results": []}
            seeds = self.pending["seeds"]
            results = self.pending["results"]
            for candidate, vector in enumerate(self.population[len(results) :], len(results) + 1):

                def frame(row, candidate=candidate):
                    on_frame(
                        {
                            **row,
                            "generation": self.completed + 1,
                            "candidate": candidate,
                            "population": self.config.population,
                        }
                    )

                frame.enabled = getattr(on_frame, "enabled", lambda: True)
                if hasattr(on_frame, "should_emit"):
                    frame.should_emit = on_frame.should_emit

                result = self.evaluate_candidate(
                    vector, seeds, control, frame if on_frame else None
                )
                if result is None:
                    return
                results.append(result)
            fitness = np.asarray([r["reward"] for r in results])
            best_index = max(range(len(results)), key=lambda i: evaluation_rank(results[i]))
            best = self.evaluate_candidate(self.population[best_index], [10001, 10002], control)
            if best is None:
                return
            champion = None
            if self.completed:
                champion = evaluate(
                    self.model,
                    self.config.size,
                    [10001, 10002],
                    self.config.max_steps,
                    control,
                    env_options=self.config.env_options(),
                )
                if champion is None:
                    return
                self.champion_reward = champion["reward"]
                self.champion_rank = evaluation_rank(champion)
            if evaluation_rank(best) > self.champion_rank:
                vector_to_parameters(self.population[best_index].clone(), self.model.parameters())
                self.champion_reward = best["reward"]
                self.champion_rank = evaluation_rank(best)
                champion = best
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
                    "evaluation_occupancy": champion["occupancy"],
                    "evaluation_win_rate": champion["win_rate"],
                    "evaluation_reasons": champion["reasons"],
                    "occupancy": max(r["max_occupancy"] for r in results),
                    "frame": champion["frame"],
                }
            )

    def state_dict(self):
        return {
            "version": 2,
            "architecture": architecture(self.config.size, self.config.observation_mode),
            "algorithm": "evolution",
            "config": self.config.to_dict(),
            "model": self.model.state_dict(),
            "population": self.population,
            "completed": self.completed,
            "champion_reward": self.champion_reward,
            "champion_rank": list(self.champion_rank),
            "rng": self.rng.bit_generator.state,
            "pending": copy.deepcopy(self.pending),
            "torch_rng": torch.get_rng_state(),
        }

    def load_state_dict(self, state):
        previous = type(self.config).from_dict(state["config"])
        if any(
            getattr(previous, key) != getattr(self.config, key)
            for key in ("algorithm", "size", "observation_mode")
        ):
            raise ValueError("для продолжения нужны прежний алгоритм, поле и наблюдение")
        if state["population"].shape != self.population.shape:
            raise ValueError("для продолжения нужна прежняя численность популяции")
        self.model.load_state_dict(state["model"])
        self.population = state["population"].clone()
        self.completed = state["completed"]
        self.champion_reward = state["champion_reward"]
        self.champion_rank = tuple(state.get("champion_rank", (-1.0, -1.0, -1.0)))
        self.pending = copy.deepcopy(state.get("pending"))
        self.rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
