import pytest
import torch

from rl_fun.snake.gpu.checkpoint import load_model, save_model
from rl_fun.snake.gpu.model import ConvActorCritic
from rl_fun.snake.gpu.vec_env import VecSnake


def observations(size=8, count=6):
    vec = VecSnake(size, count, "cpu", seed=0)
    return vec.observation(), vec.action_mask()


def test_forward_returns_logits_and_values_for_the_batch():
    obs, _ = observations()
    model = ConvActorCritic(8)

    logits, value = model(obs)

    assert tuple(logits.shape) == (6, 3) and tuple(value.shape) == (6,)


def test_greedy_actions_are_deterministic():
    obs, _ = observations()
    model = ConvActorCritic(8).eval()

    assert torch.equal(model.greedy(obs), model.greedy(obs))


def test_greedy_never_picks_a_masked_action():
    obs, _ = observations(count=64)
    mask = torch.rand(64, 3) < 0.5
    mask[:, 1] = True
    model = ConvActorCritic(8).eval()

    actions = model.greedy(obs, mask)

    assert bool(mask.gather(1, actions[:, None]).all())


def test_larger_boards_have_larger_inputs_and_parameter_counts():
    assert ConvActorCritic(16).parameter_count() > ConvActorCritic(8).parameter_count()


def test_checkpoint_round_trip_preserves_the_weights(tmp_path):
    model = ConvActorCritic(8)
    path = tmp_path / "model.pt"

    save_model(path, model, {"note": "test"})
    loaded, meta = load_model(path, size=8)

    assert meta["note"] == "test"
    assert all(
        torch.equal(a, b)
        for a, b in zip(model.state_dict().values(), loaded.state_dict().values(), strict=True)
    )


def test_checkpoint_for_another_board_size_is_rejected(tmp_path):
    save_model(tmp_path / "model.pt", ConvActorCritic(8), {})

    with pytest.raises(ValueError, match="размер"):
        load_model(tmp_path / "model.pt", size=12)


def test_checkpoint_with_unknown_version_is_rejected(tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"format": "snake-gpu", "version": 99, "size": 8}, path)

    with pytest.raises(ValueError, match="версия"):
        load_model(path)


def test_checkpoint_with_non_finite_weights_is_rejected(tmp_path):
    model = ConvActorCritic(8)
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    save_model(tmp_path / "model.pt", model, {})

    with pytest.raises(ValueError, match="некорректные"):
        load_model(tmp_path / "model.pt")


def test_checkpoint_loading_refuses_arbitrary_objects(tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"format": "snake-gpu", "version": 1, "size": 8, "model": ConvActorCritic(8)}, path)

    with pytest.raises(ValueError):
        load_model(path)


def test_saving_is_atomic_and_leaves_no_temporary_files(tmp_path):
    save_model(tmp_path / "model.pt", ConvActorCritic(8), {})

    assert [item.name for item in tmp_path.iterdir()] == ["model.pt"]
