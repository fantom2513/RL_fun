# Snake: GPU training and a pretrained 16x16 completion model

Date: 2026-10-16
Status: design, awaiting review

## Intent

The user wants a Snake agent that wins the game (fills every cell) on 16x16, as a
showpiece next to the sandbox algorithms. The user approved:

- training on the local GPU (RTX 5070, 12 GB) and switching torch to the CUDA 12.8 build;
- a Hamiltonian-cycle expert as a teacher: imitation pretraining, then PPO fine-tuning.

The existing sandbox algorithms (DQN, TD3, evolution, the CPU PPO) stay as they are.

## Why the current runs fail

`docs/snake-research.md` records about 3 million environment steps in total, 20 trials,
no full win, mean occupancy 1-12%. Measured throughput is roughly 270 steps/s
(249,056 PPO steps in 900 s). The sources report orders of magnitude more experience:
[snake-ai](https://github.com/linyiLYi/snake-ai/blob/master/main/train_cnn.py) uses
100M steps on 12x12, and [Head-centered PPO](https://elichen.github.io/snake/index.html)
uses 256 parallel environments on 20x20 (about 81% wins, author-reported, steps not
stated). A pure-RL search on 16x16 needs about 1000x the current throughput, and even
then success is not guaranteed: a masked PPO run with 100M steps never filled a 4x4
board, and [AlphaSnake](https://arxiv.org/abs/2211.09622) needed tree search to pass 50%.
Following a Hamiltonian cycle never loses, but RL does not discover it unaided. So the
design combines a fast environment, a teacher that always wins, and RL to remove the
teacher's weaknesses.

## Honesty rules

- The final model is a neural network. At evaluation it plays greedily, with no teacher,
  no action mask and no safety shield. Masked results may be reported only as secondary.
- It is described as "pretrained from a Hamiltonian expert, fine-tuned with PPO", never
  as learned from scratch.
- Its observation includes a cycle-order plane (below). This is disclosed in the model
  card and in `docs/snake-research.md`.
- Acceptance is measured on the authoritative CPU `SnakeEnv`, not on the GPU copy.

## Acceptance

16x16, three starting segments, 253 foods. At least 90 wins out of 100 games on the
frozen validation seeds `3000000..3000099` (`presets.benchmark_seeds()`), greedy play,
CPU `SnakeEnv`. This matches the existing `verification_status` threshold. Tuning uses
the 20 tuning seeds `1000000..1000019` only. Smaller boards (8, 12) are a ladder for
debugging and do not satisfy acceptance. If the target is not reached, the result is
reported as it is, with the numbers; the preset stays "experimental".

## Components

All new code lives in `src/rl_fun/snake/gpu/`; each file has one job.

1. `cycle.py`. `hamiltonian_cycle(size)` returns an order grid: the cycle index of each
   cell. Even sizes only (a grid cycle needs an even side); odd sizes raise `ValueError`.
2. `vec_env.py`. `VecSnake(size, num_envs, device, seed)`: B games as torch tensors.
   State per game: a `body` grid of ints (cell value = steps until it is vacated; head
   = length, tail = 1, empty = 0), head position, direction, length, food, steps since
   food. One step is a few batched tensor operations over `(B, N, N)`. Same rules as
   `SnakeEnv`: three relative actions, the tail cell is free unless eating, growth on
   food, win on full occupancy, truncation after `starvation_factor * N^2` steps without
   food. Finished games auto-reset; the final outcome is returned alongside.
   Reset modes: standard start, and a start on a contiguous segment of the cycle of
   random length (a state the teacher can continue from). Food placement samples free
   cells with a seeded `torch.Generator`. Rewards: `+food_reward`, `-death_penalty`,
   `-step_penalty`, `+win_reward`, the same parameters as `SnakeConfig`.
3. Observation (shared contract). Six planes of the `(2N-1) x (2N-1)` window centred on
   the head and rotated so the heading is up: head, body order (rank/length), food,
   length/N^2 (constant plane), out-of-bounds, and **cycle phase**
   `((cycle[cell] - cycle[head]) mod N^2) / N^2`. Implemented in `VecSnake` and, as the
   new mode `observation_mode="egocentric_cycle"`, in `SnakeEnv`, so the model can be
   evaluated by the authoritative environment. A test requires both to agree exactly on
   random states. An action mask (`True` for non-fatal moves) is exposed by `VecSnake`.
4. `teacher.py`. A vectorized Hamiltonian-cycle expert with shortcuts. Candidate moves
   are the three relative actions that are non-fatal. A shortcut move is allowed when
   its cycle distance from the head does not pass the food and stays behind the tail's
   cycle position with a safety margin, and shortcuts are disabled once the snake is
   longer than a configured fraction of the board. Among the allowed moves it picks the
   one closest to the food along the cycle; following the cycle (distance 1) is always
   allowed, so the teacher never gets stuck. `shortcuts=False` gives the pure cycle
   follower. The expert is valid only from states whose body follows the cycle order,
   which holds for the standard start and for cycle-segment starts.
5. `model.py`. `ConvActorCritic`: a small CNN over the six planes (translation
   invariance helps the window), a scalar branch for length, a 3-way policy head and a
   value head. Parameter budget of a few million. Checkpoints use the existing
   tensor-only atomic save (`weights_only=True` loading) with a new version number.
6. `imitation.py`. Behaviour cloning: teacher rollouts on the GPU produce
   `(observation, action)` pairs, cross-entropy training, accuracy and greedy win-rate
   reports. Plain cloning without DAgger, since the teacher's guarantee does not hold
   off its own state distribution.
7. `ppo.py`. Batched PPO on `VecSnake`: GAE with gamma 0.999, clipped objective,
   optional invalid-move mask in training, entropy bonus, best-checkpoint tracking on
   greedy evaluation. Starts from the cloned weights. A critic-only warm-up phase fits
   the value head before the policy is unfrozen, and an optional KL penalty to the
   cloned policy limits early drift. Curriculum starts mix standard starts with long
   bodies (cycle segments, later arbitrary connected bodies) and evaluation always uses
   the standard start.
8. `evaluate.py`. Greedy evaluation of a checkpoint on a given seed list, on either
   backend (GPU `VecSnake` for speed, CPU `SnakeEnv` for acceptance). Reports wins,
   occupancy, steps and failure reasons.
9. Scripts: `scripts/snake_gpu_bench.py` (throughput), `scripts/snake_gpu_train.py`
   (stages: teacher check, clone, fine-tune, ladder 8 -> 12 -> 16),
   `scripts/snake_gpu_verify.py` (acceptance run on CPU `SnakeEnv`).

## Data flow

teacher on `VecSnake` -> `(obs, action)` dataset -> `ConvActorCritic` cloned ->
PPO on `VecSnake` -> checkpoint -> CPU `SnakeEnv` acceptance run. Logs are JSON lines
under `runs/snake-gpu/`; checkpoints are never committed.

## Targets and gates

- Throughput gate: at least 100,000 environment steps/s on 16x16 with 4096 games,
  observations included. If lower, profile and batch more before going on.
- Teacher gate: the pure cycle follower wins 100% of games on 8, 12 and 16; the
  shortcut teacher also wins 100% on at least 1000 games (8x8) and 200 games (16x16)
  with fewer steps per game than the pure follower.
- Cloning gate (8x8, then 16x16): report greedy win rate of the clone on the tuning
  seeds. A clone below 50% on 8x8 means the model or observation is wrong; stop and fix
  before touching PPO.
- Fine-tuning gate: PPO must not reduce the win rate of the cloned model on the tuning
  seeds. If it does, lower the learning rate or raise the KL weight.

## Dependencies

`pyproject.toml` switches the torch index from `pytorch-cpu` to
`https://download.pytorch.org/whl/cu128` (RTX 5070 is Blackwell, `sm_120`, which needs
CUDA 12.8 or newer). No other new packages. The CUDA wheel also runs on CPU, so the
other tests are unaffected. If `torch.cuda.is_available()` is false, GPU tests skip and
training scripts exit with a clear message.

## Error handling

Invalid parameters (odd board, sizes outside 8-32, non-positive batch) raise
`ValueError` with a clear message. A checkpoint with a different board size, plane count
or version is rejected at load. Training scripts write the checkpoint atomically and
resume from the latest one. Non-finite losses stop training and report the step.

## Testing

TDD for all new logic; tests parallel the sources in `tests/snake/gpu/`.

- Cycle: each cell once, neighbours adjacent, closing edge adjacent, odd size rejected.
- Dynamics equivalence: scripted action sequences with food positions forced to be
  identical give identical snake, score, reward and termination on `VecSnake` and
  `SnakeEnv`; covers tail-chase, eating the cell the tail leaves, wall and body
  collision, win on a tiny board, starvation truncation, auto-reset.
- Observation equivalence between `VecSnake` and `SnakeEnv` on random states.
- Reproducibility: the same seed gives the same trajectories.
- Teacher gates above; model shapes; checkpoint round trip; GAE against a hand
  calculation; a PPO update on a toy problem; clone accuracy on a tiny run.
- GPU tests are marked and skipped without CUDA; the logic is also tested on CPU
  tensors, so the suite stays green on any machine.

## Out of scope

Changes to DQN, TD3, evolution or the CPU PPO; new lab screens (the final step adds
only a preset entry and playback of the pretrained model through the existing
evaluation path); odd boards; multi-GPU; the flood-fill auxiliary head and a recurrent
policy (contingencies if fine-tuning stalls, to be raised before being built).

## Operational rules

Every long training run (more than a few minutes) is started only after the user
confirms it. Results and failures go into `docs/snake-research.md`.
