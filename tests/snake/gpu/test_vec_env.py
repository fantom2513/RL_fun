import pytest
import torch
from rl_fun.snake.gpu.vec_env import REASONS, VecSnake

from rl_fun.snake.env import DIRECTIONS, SnakeEnv
from rl_fun.snake.gpu.cycle import hamiltonian_cycle

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def act(vec, *actions):
    return torch.tensor(actions, dtype=torch.long, device=vec.device)


def row_path(size):
    return [
        (x, y)
        for y in range(size)
        for x in (range(size) if y % 2 == 0 else reversed(range(size)))
    ]


def invariants_hold(vec):
    body = vec.body.view(vec.num_envs, -1)
    for i in range(vec.num_envs):
        cells = vec.snake_cells(i)
        length = int(vec.length[i])
        values = sorted(body[i][body[i] > 0].tolist())
        if values != list(range(1, length + 1)) or len(cells) != length:
            return False
        if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1 for a, b in zip(cells, cells[1:], strict=False)):
            return False
        food = tuple(vec.food[i].tolist())
        if length < vec.size**2 and (food in cells or not _inside(food, vec.size)):
            return False
    return True


def _inside(cell, size):
    return 0 <= cell[0] < size and 0 <= cell[1] < size


def make_pair(size=8, seed=0, device="cpu", **options):
    env = SnakeEnv(size, **options)
    env.reset(seed=seed)
    vec = VecSnake(size, 1, device, seed, **options)
    vec.load_snapshot(0, env.snake, env.food, env.direction)
    return env, vec


def step_pair(env, vec, action):
    """Step both environments; mirror the reference's new food so the games stay identical."""
    before = env.score
    _, reward, terminated, truncated, _ = env.step(action)
    v_reward, v_terminated, v_truncated, info = vec.step(act(vec, action))
    assert float(v_reward[0]) == pytest.approx(reward)
    assert bool(v_terminated[0]) == terminated
    assert bool(v_truncated[0]) == truncated
    done = terminated or truncated
    if done:
        assert REASONS[int(info["reason"][0])] == env.reason
        assert int(info["score"][0]) == env.score
        assert int(info["length"][0]) == len(env.snake)
        assert int(info["steps"][0]) == env.steps
    elif env.score != before:
        vec.set_food(0, env.food)
    if not done:
        assert vec.snake_cells(0) == [tuple(cell) for cell in env.snake]
        assert tuple(vec.food[0].tolist()) == env.food
        assert int(vec.direction[0]) == env.direction
    return done


def test_standard_start_matches_the_reference_environment():
    env = SnakeEnv(8)
    env.reset(seed=0)
    vec = VecSnake(8, 3, "cpu", seed=0)

    assert vec.snake_cells(0) == [tuple(cell) for cell in env.snake]
    assert int(vec.direction[0]) == env.direction == 0


def test_straight_move_shifts_the_snake_without_growing():
    env, vec = make_pair()
    env.food = (0, 0)
    vec.set_food(0, (0, 0))

    done = step_pair(env, vec, 0)

    assert not done
    assert len(vec.snake_cells(0)) == 3


def test_turns_follow_the_relative_actions():
    env, vec = make_pair()
    env.food = (0, 0)
    vec.set_food(0, (0, 0))

    for action in (1, 2, 2, 0, 1):
        step_pair(env, vec, action)


def test_eating_grows_the_snake_and_scores():
    env, vec = make_pair()
    head = env.snake[0]
    env.food = (head[0] + 1, head[1])
    vec.set_food(0, env.food)

    step_pair(env, vec, 0)

    assert len(vec.snake_cells(0)) == 4
    assert int(vec.score[0]) == 1


def test_moving_into_the_cell_the_tail_leaves_is_allowed():
    env, vec = make_pair()
    env.snake = [(3, 4), (4, 4), (4, 3), (3, 3)]  # heading left; the tail (3,3) is above the head
    env.direction = DIRECTIONS.index((-1, 0))
    env.food = (0, 0)
    vec.load_snapshot(0, env.snake, env.food, env.direction)

    done = step_pair(env, vec, 2)  # a right turn heads up into the cell the tail vacates

    assert not done

def test_wall_collision_ends_the_game_with_the_death_penalty():
    env, vec = make_pair()
    env.snake = [(7, 4), (6, 4), (5, 4)]
    env.direction = 0
    env.food = (0, 0)
    vec.load_snapshot(0, env.snake, env.food, env.direction)

    done = step_pair(env, vec, 0)

    assert done


def test_self_collision_ends_the_game():
    env, vec = make_pair()
    env.snake = [(3, 3), (4, 3), (4, 4), (3, 4), (2, 4), (1, 4)]  # heading left, hooked body
    env.direction = DIRECTIONS.index((-1, 0))
    env.food = (0, 0)
    vec.load_snapshot(0, env.snake, env.food, env.direction)

    done = step_pair(env, vec, 1)  # a left turn heads down into (3,4), a middle segment

    assert done and env.reason == "collision"

def test_filling_the_board_is_a_win_with_the_win_reward():
    size = 8
    path = row_path(size)
    env, vec = make_pair(size, win_reward=100.0)
    env.snake = list(reversed(path[:-1]))
    env.food = path[-1]
    env.direction = DIRECTIONS.index((path[-2][0] - path[-3][0], path[-2][1] - path[-3][1]))
    vec.load_snapshot(0, env.snake, env.food, env.direction)

    done = step_pair(env, vec, 0)

    assert done and env.reason == "win"


def test_going_without_food_too_long_truncates_the_game():
    env, vec = make_pair(starvation_factor=1)
    env.food = (0, 0)
    vec.set_food(0, (0, 0))

    results = [step_pair(env, vec, 1) for _ in range(8 * 8)]

    assert results[-1] and not any(results[:-1]) and env.reason == "starvation"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", (8, 9, 12))
def test_random_games_stay_identical_to_the_reference(size, device):
    generator = torch.Generator().manual_seed(size)
    for seed in range(30):
        env, vec = make_pair(size, seed, device)
        for _ in range(400):
            action = int(torch.randint(3, (1,), generator=generator))
            if step_pair(env, vec, action):
                break


@pytest.mark.parametrize("device", DEVICES)
def test_random_batches_keep_the_state_consistent(device):
    vec = VecSnake(8, 64, device, seed=3)
    generator = torch.Generator(device=device).manual_seed(1)

    for _ in range(300):
        actions = torch.randint(3, (64,), generator=generator, device=device)
        vec.step(actions)
        assert invariants_hold(vec)


def test_same_seed_gives_the_same_trajectories():
    def run():
        vec = VecSnake(8, 16, "cpu", seed=5)
        generator = torch.Generator().manual_seed(2)
        rewards = []
        for _ in range(100):
            reward, *_ = vec.step(torch.randint(3, (16,), generator=generator))
            rewards.append(reward.clone())
        return torch.stack(rewards), vec.body.clone()

    first, second = run(), run()

    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


def test_finished_games_restart_automatically_with_a_fresh_snake():
    vec = VecSnake(8, 4, "cpu", seed=0)
    vec.load_snapshot(0, [(7, 4), (6, 4), (5, 4)], (0, 0), 0)

    _, terminated, _, info = vec.step(act(vec, 0, 0, 0, 0))

    assert bool(terminated[0]) and bool(info["done"][0])
    assert vec.snake_cells(0) == [(4, 4), (3, 4), (2, 4)]
    assert int(vec.score[0]) == 0 and int(vec.steps[0]) == 0


def test_cycle_segment_starts_follow_the_cycle_order():
    vec = VecSnake(8, 200, "cpu", seed=1, start_mode="cycle_segment", max_start_length=40)
    order = hamiltonian_cycle(8)

    lengths = vec.length.tolist()
    for i in range(vec.num_envs):
        cells = vec.snake_cells(i)  # head first
        indexes = [int(order[y, x]) for x, y in cells]
        assert all((a - b) % 64 == 1 for a, b in zip(indexes, indexes[1:], strict=False))
    assert min(lengths) >= 3 and max(lengths) <= 40 and len(set(lengths)) > 10
    assert invariants_hold(vec)


def test_cycle_segment_starts_need_an_even_board():
    with pytest.raises(ValueError):
        VecSnake(9, 4, "cpu", seed=0, start_mode="cycle_segment")


@pytest.mark.parametrize(
    "arguments",
    [
        {"size": 3, "num_envs": 4},
        {"size": 8, "num_envs": 0},
        {"size": 8, "num_envs": 4, "start_mode": "random"},
        {"size": 8, "num_envs": 4, "starvation_factor": 0},
    ],
)
def test_invalid_parameters_are_rejected(arguments):
    with pytest.raises(ValueError):
        VecSnake(device="cpu", seed=0, **arguments)
