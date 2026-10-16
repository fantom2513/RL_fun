"""Batched PPO on ``VecSnake`` with an optional move mask, critic warm-up and KL anchor."""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from rl_fun.snake.gpu.model import ConvActorCritic
from rl_fun.snake.gpu.vec_env import VecSnake

_BLOCKED = -1e9


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 2
    minibatch: int = 8192
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    kl_coef: float = 0.0
    max_grad_norm: float = 0.5
    use_mask: bool = True


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Remove masked moves; a game with no safe move keeps its logits (all moves are lost)."""
    return torch.where(mask.any(dim=1, keepdim=True), logits.masked_fill(~mask, _BLOCKED), logits)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantages and returns for ``(steps, games)`` tensors.

    A finished game cuts the bootstrap; truncated games are treated as finished.
    """
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(last_value)
    next_value = last_value
    for step in reversed(range(rewards.shape[0])):
        alive = (~dones[step]).float()
        delta = rewards[step] + gamma * alive * next_value - values[step]
        carry = delta + gamma * gae_lambda * alive * carry
        advantages[step] = carry
        next_value = values[step]
    return advantages, advantages + values


def policy_loss(
    log_prob: torch.Tensor, old_log_prob: torch.Tensor, advantage: torch.Tensor, clip: float
) -> torch.Tensor:
    ratio = (log_prob - old_log_prob).exp()
    clipped = ratio.clamp(1 - clip, 1 + clip)
    return -torch.minimum(ratio * advantage, clipped * advantage).mean()


def kl_divergence(logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    """Mean KL(reference || policy) over the batch."""
    reference = reference_logits.log_softmax(dim=1)
    return (reference.exp() * (reference - logits.log_softmax(dim=1))).sum(dim=1).mean()


@torch.no_grad()
def collect_rollout(
    model: ConvActorCritic, vec: VecSnake, steps: int, use_mask: bool = True
) -> dict[str, torch.Tensor]:
    """Play ``steps`` steps in every game with the sampling policy and record them."""
    model.eval()
    record: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("observations", "actions", "log_probs", "values", "rewards", "dones")
    }
    record["masks"] = []
    finished = wins = 0
    for _ in range(steps):
        observation, mask = vec.observation(), vec.action_mask()
        logits, value = model(observation)
        logits = masked_logits(logits, mask) if use_mask else logits
        log_probs = logits.log_softmax(dim=1)
        actions = torch.multinomial(log_probs.exp(), 1, generator=vec.generator)[:, 0]
        reward, terminated, truncated, info = vec.step(actions)
        record["observations"].append(observation)
        record["masks"].append(mask)
        record["actions"].append(actions)
        record["log_probs"].append(log_probs.gather(1, actions[:, None])[:, 0])
        record["values"].append(value)
        record["rewards"].append(reward)
        record["dones"].append(terminated | truncated)
        finished += int(info["done"].sum())
        wins += int((info["done"] & (info["reason"] == 2)).sum())
    rollout = {key: torch.stack(items) for key, items in record.items()}
    rollout["last_value"] = model(vec.observation())[1]
    rollout["finished"], rollout["wins"] = finished, wins
    return rollout


def finish_rollout(rollout: dict, config: PPOConfig) -> dict:
    advantages, returns = compute_gae(
        rollout["rewards"],
        rollout["values"],
        rollout["dones"],
        rollout["last_value"],
        config.gamma,
        config.gae_lambda,
    )
    return {**rollout, "advantages": advantages, "returns": returns}


def ppo_update(
    model: ConvActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: dict,
    config: PPOConfig,
    reference: ConvActorCritic | None = None,
    critic_only: bool = False,
) -> dict[str, float]:
    """Run ``config.epochs`` passes of minibatch PPO over a finished rollout."""
    model.train()
    flat = {
        key: rollout[key].flatten(0, 1)
        for key in ("observations", "actions", "log_probs", "advantages", "returns", "masks")
    }
    advantages = flat["advantages"]
    flat["advantages"] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    count = flat["actions"].shape[0]
    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "kl": 0.0,
        "clip_fraction": 0.0,
    }
    updates = 0
    for _ in range(config.epochs):
        order = torch.randperm(count, device=flat["actions"].device)
        for start in range(0, count, config.minibatch):
            index = order[start : start + config.minibatch]
            batch = {key: value[index] for key, value in flat.items()}
            logits, value = model(batch["observations"])
            if config.use_mask:
                logits = masked_logits(logits, batch["masks"])
            log_probs = logits.log_softmax(dim=1)
            log_prob = log_probs.gather(1, batch["actions"][:, None])[:, 0]
            actor = policy_loss(log_prob, batch["log_probs"], batch["advantages"], config.clip)
            critic = functional.mse_loss(value, batch["returns"])
            entropy = -(log_probs.exp() * log_probs).sum(dim=1).mean()
            kl = torch.zeros((), device=logits.device)
            if reference is not None and config.kl_coef:
                with torch.no_grad():
                    reference_logits = reference(batch["observations"])[0]
                    if config.use_mask:
                        reference_logits = masked_logits(reference_logits, batch["masks"])
                kl = kl_divergence(logits, reference_logits)
            if critic_only:
                loss = config.value_coef * critic
            else:
                loss = (
                    actor
                    + config.value_coef * critic
                    - config.entropy_coef * entropy
                    + config.kl_coef * kl
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            optimizer.step()
            ratio = (log_prob - batch["log_probs"]).exp()
            totals["policy_loss"] += float(actor.detach())
            totals["value_loss"] += float(critic.detach())
            totals["entropy"] += float(entropy.detach())
            totals["kl"] += float(kl.detach())
            totals["clip_fraction"] += float(((ratio - 1).abs() > config.clip).float().mean())
            updates += 1
    return {key: value / updates for key, value in totals.items()}
