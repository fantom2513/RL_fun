"""Algorithm dispatch shared by interactive training and benchmarks."""


def create_trainer(config):
    if config.algorithm == "td3":
        from rl_fun.snake.td3 import TD3Trainer

        return TD3Trainer(config)
    if config.algorithm == "evolution":
        from rl_fun.snake.evolution import EvolutionTrainer

        return EvolutionTrainer(config)
    if config.algorithm in ("dqn", "double_dqn"):
        from rl_fun.snake.dqn import DQNTrainer

        return DQNTrainer(config)
    from rl_fun.snake.on_policy import OnPolicyTrainer

    return OnPolicyTrainer(config)
