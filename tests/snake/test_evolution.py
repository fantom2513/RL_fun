import numpy as np
import torch

from rl_fun.snake.config import SnakeConfig
from rl_fun.snake.evolution import EvolutionTrainer
from rl_fun.snake.policy import Policy, evaluate


def test_greedy_cycle_ends_early_with_failure_penalty():
    model = Policy()
    for parameter in model.parameters():
        parameter.data.zero_()
    model[-1].bias.data[1] = 10  # always turn left: repeats a square without food
    frames = []
    result = evaluate(model, 8, [42], 1000, on_frame=frames.append)
    assert result["frame"]["steps"] < 10
    assert result["frame"]["done"] and result["frame"]["reason"] == "loop"
    assert result["score"] == 0 and result["reward"] < -10
    assert frames[-1]["ended"]


def test_cycling_population_still_completes_generations_promptly():
    trainer = EvolutionTrainer(
        SnakeConfig(algorithm="evolution", size=8, population=4, generations=2, max_steps=1000)
    )
    trainer.population.zero_()
    trainer.population[:, -2] = 10  # final layer action-left bias
    frames, rows = [], []
    trainer.run(lambda: True, rows.append, frames.append)
    assert trainer.completed == 2 and len(rows) == 2
    first_generation = [row for row in frames if row["generation"] == 1]
    assert max(row["frame"]["steps"] for row in first_generation) < 10
    assert {row["candidate"] for row in first_generation} == {1, 2, 3, 4}


def test_resume_recomputes_champion_fitness_instead_of_trusting_old_reward():
    config = SnakeConfig(
        algorithm="evolution",
        size=8,
        population=4,
        generations=2,
        max_steps=20,
        evaluation_episodes=1,
    )
    previous = EvolutionTrainer(config)
    previous.completed = 1
    previous.champion_reward = 1e9  # checkpoint produced with a different scoring rule
    resumed = EvolutionTrainer(config)
    resumed.load_state_dict(previous.state_dict())
    resumed.run(lambda: True, lambda _: None)
    actual = evaluate(resumed.model, 8, [10001, 10002], 20)["reward"]
    assert resumed.champion_reward == actual


def test_elites_survive_and_offspring_mutate():
    trainer = EvolutionTrainer(
        SnakeConfig(
            algorithm="evolution",
            population=4,
            elite_fraction=0.25,
            generations=1,
            evaluation_episodes=1,
            max_steps=20,
        )
    )
    old = trainer.population.clone()
    trainer.reproduce(np.asarray([0.0, 10.0, 2.0, 3.0]))
    torch.testing.assert_close(trainer.population[0], old[1])
    assert all(not torch.equal(row, old[1]) for row in trainer.population[1:])


def test_training_and_state_roundtrip():
    config = SnakeConfig(
        algorithm="evolution", population=4, generations=2, evaluation_episodes=1, max_steps=20
    )
    trainer = EvolutionTrainer(config)
    rows = []
    trainer.run(lambda: True, rows.append)
    assert len(rows) == 2 and trainer.completed == 2
    assert rows[-1]["best_reward"] >= rows[-1]["mean_reward"]
    restored = EvolutionTrainer(config)
    restored.load_state_dict(trainer.state_dict())
    torch.testing.assert_close(restored.population, trainer.population)
    torch.testing.assert_close(restored.model(torch.zeros(11)), trainer.model(torch.zeros(11)))
    assert restored.rng.random() == trainer.rng.random()


def test_evaluation_seeds_are_shared_and_stop_is_cooperative():
    trainer = EvolutionTrainer(
        SnakeConfig(algorithm="evolution", population=4, generations=1, max_steps=20)
    )
    calls = []

    def control():
        calls.append(1)
        return len(calls) < 3

    trainer.run(control, lambda _: None)
    assert trainer.completed == 0 and len(calls) == 3
    seeds = trainer.generation_seeds()
    assert len(seeds) == trainer.config.evaluation_episodes
    assert (
        trainer.evaluate_candidate(trainer.population[0], seeds)["reward"]
        == (trainer.evaluate_candidate(trainer.population[0], seeds)["reward"])
    )


def test_mid_generation_checkpoint_keeps_seeds_and_results():
    config = SnakeConfig(
        algorithm="evolution", population=4, generations=2, evaluation_episodes=1, max_steps=20
    )
    complete = EvolutionTrainer(config)
    expected = []
    complete.run(lambda: True, expected.append)
    interrupted = EvolutionTrainer(config)
    calls = 0

    def control():
        nonlocal calls
        calls += 1
        return calls < 25

    first = []
    interrupted.run(control, first.append)
    restored = EvolutionTrainer(config)
    restored.load_state_dict(interrupted.state_dict())
    rest = []
    restored.run(lambda: True, rest.append)
    assert first + rest == expected
    torch.testing.assert_close(restored.population, complete.population)
