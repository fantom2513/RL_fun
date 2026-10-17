# Snake completion research and experiments

The target is complete occupancy on at least 16×16, starting with three
segments. A full game collects 253 food items. Mean occupancy is a diagnostic,
not a completion rate. No learned checkpoint has established this target yet.

## Primary sources

- [Head-centered PPO](https://elichen.github.io/snake/index.html): the author
  reports strong 20×20 completion after head-centered observations, curriculum
  and continued PPO training. This is an author report, not a result reproduced
  by this repository. The published architecture description also needs source
  verification before a faithful reproduction can be claimed.
- [Curriculum Learning with Snake](https://web.stanford.edu/class/aa228/reports/2020/final16.pdf):
  CNN-DQN with gradually reduced food density and collision forgiveness. The
  report discusses the mismatch between modified training and normal games.
- [AlphaSnake](https://arxiv.org/abs/2211.09622): Monte Carlo tree search combined
  with RL policy iteration. This is a different inference method, not evidence
  that a short DQN/PPO parameter sweep should complete 16×16.
- [snake-ai](https://github.com/linyiLYi/snake-ai): an implementation comparing
  CNN and MLP observations. Its [training code](https://github.com/linyiLYi/snake-ai/blob/master/main/train_cnn.py)
  uses MaskablePPO, a 100-million-step budget, and 32 subprocess environments
  (64 for the MPS configuration). The default board is 12×12, RGB observations
  are resized to 84×84, and actions are four absolute directions. Its
  [evaluation](https://github.com/linyiLYi/snake-ai/blob/master/main/test_cnn.py)
  also masks immediately fatal moves. Those weights and reported scores cannot
  be treated as a directly compatible pure-policy 16×16 solution here.

## Current experiments

All screening uses the same 20 tuning seeds and ordinary starts. Independent
100-game validation is reserved for a frozen candidate after selection.
Training caps and actual steps are recorded because runs have different speeds.

The first 14 experiments added 1,871,056 environment steps, with no full wins.
Their records are in `runs/lab/snake/search-round1.json`, `search-round2.json`
and `search-round3.json`. Older checkpoints used larger training budgets, so
this is not a matched-budget comparison of algorithms.

The next continuation experiments used a learning rate of 0.0001:

| Experiment | Added steps | Mean occupancy | Full wins | Training seconds |
| --- | ---: | ---: | ---: | ---: |
| PPO, gamma 0.999, entropy 0.005 | 249,056 | 11.80% | 0/20 | 900.03 |
| Double DQN, epsilon floor 0.01 | 241,744 | 10.80% | 0/20 | 900.00 |

Saved checkpoints and detailed evaluation outcomes are referenced by
`runs/lab/snake/search-research-continuations.json`. The PPO result improves
the screening score but does not meet the completion target.

Two fresh PPO experiments compare head-centered observations with and without
long-body training resets. This experimental curriculum mixes 50% ordinary
starts with connected synthetic bodies; the maximum starting length increases
from 25% to 95% of the board over 250,000 steps. It supplies no expert actions.
This reset schedule is a hypothesis, not the curriculum reproduced from either
source. Evaluation and playback retain the standard length-three start.

Both runs stopped after 900 seconds: the ordinary variant added 141,888 steps,
and the curriculum variant added 140,844. Each reached 1.17% mean occupancy and
0/20 full wins. This limited experiment did not demonstrate an improvement
from the new observation or the long-body reset curriculum.

Early trajectories in both variants predominantly looped without food when
step rewards were zero. A follow-up configuration increases entropy to 0.03,
uses gamma 0.99, a step cost of 0.001, and an additional 0.01 cost after 256
consecutive moves without food. Two otherwise identical ordinary-start runs
compare `distance_reward=0` against `distance_reward=0.05`. This adds/subtracts
the configured value per cell approached/retreated from the previous food;
respawning food does not create progress reward. Collisions keep their original
penalty. This is heuristic reward shaping, with a risk of greedy self-trapping.
Configurations and results are stored separately in `search-spatial*.json`;
the reward change must be evaluated rather than assumed beneficial.

The ordinary-start reward comparison also failed to establish completion:
the control added 184,215 steps and reached 1.25% mean occupancy; the distance
reward variant added 183,296 steps and reached 1.17%. Both recorded 0/20 wins
and loop endings in all tuning games. Each run used 900 seconds of training.

The next comparison resumes the strongest board-observation PPO checkpoint,
with GAE lambda 0.99, entropy 0.01, step cost 0.001, and a small idle cost after
1,024 steps without food. It compares distance reward 0 against 0.02. Each run
has a budget of two million added steps, capped at 1,200 training seconds;
actual budgets and saved checkpoints are recorded in `search-board-credit.json`.

A third run resumes the same previously trained board PPO with the distance
reward setting and `curriculum_steps=2000000`. This tests long-body resets
after food-seeking has already been learned, rather than combining a new
observation and an untrained policy. It has a one-million-step added budget,
capped at 1,200 seconds, and records results in `search-board-curriculum.json`.
All three follow-up runs still evaluate from ordinary length-three starts.

The 20 completed screening trials before those follow-ups added 3,012,099
environment steps in total and recorded no full wins. Splitting steps between
configurations is not equivalent to training one model for that entire budget.

## GPU pipeline: Hamiltonian-cycle teacher and cloning (2026-10-16)

Design: `docs/superpowers/specs/2026-10-16-snake-gpu-design.md`. Code: `src/rl_fun/snake/gpu/`.
The batched environment runs 1.14 million steps/s on an RTX 5070 (16x16, 4096 games,
observations included), against about 270 steps/s for the earlier CPU runs.

The network is a convolutional policy over the head-centred window with a sixth plane
that gives every cell's distance ahead along a fixed Hamiltonian cycle. It is pretrained by
imitating a cycle-following teacher with safe shortcuts (the teacher wins every game it was
checked on: 1000 games on 8x8, 200 on 16x16) and is evaluated greedily, without the teacher
and without an action mask. It is not learned from scratch.

Cloning gate on 8x8 (4,000 iterations, 4 minutes, `runs/snake-gpu/clone-8`):
99.8% action accuracy; 127 of 128 greedy wins on the batched environment; 20 of 20 wins on
the reference `SnakeEnv` tuning seeds. 16x16 acceptance (90 of 100 on seeds
3000000..3000099) is not yet measured.

Cloning on 16x16 (12,000 iterations, 36 minutes, `runs/snake-gpu/clone-16`): action accuracy
99.98%, but greedy play is unstable between checkpoints (64-game batched evaluations at every
1,000 iterations: 8%, 28%, 48%, 39%, 77%, 72%, 97%, 44%, 80%, 94%, 94%, 77% wins). The final
model, evaluated on the 20 tuning seeds in the reference `SnakeEnv`, won 14 of 20 (70%, mean
occupancy 72%); the 6 failures were all starvation, i.e. the policy looped without food.
Below the 90-of-100 acceptance bar; no best-checkpoint selection and no PPO fine-tuning yet.

PPO fine-tuning stage (`scripts/snake_gpu_train.py finetune`, reward food +1, death -1, step -0.0005,
win +2, no move mask in training because a mask would hide the penalty for fatal moves). On 8x8,
starting from the 99% clone (`runs/snake-gpu/ppo-8-*`): a 5-iteration critic warm-up followed by
lr 3e-5 dropped greedy wins from 99% to 9% within 20 iterations. With a 60-iteration critic-only
warm-up the clone's play is unchanged (99.2% throughout), and after unfreezing at lr 1e-5:
without a KL anchor wins fell to 27-68%; with a KL anchor to the clone (coefficient 2.0) they
went 73%, 92%, 90%, 95%, 98% over the next 80 iterations. The 8x8 clone has no headroom, so
these runs only show stability; the best checkpoint (the clone itself) was kept in both.
Whether PPO improves the 16x16 clone is not yet known.

Pending run (not started): PPO on 16x16 from `runs/snake-gpu/clone-16/clone.pt`, command
`uv run python scripts/snake_gpu_train.py finetune --size 16 --resume runs/snake-gpu/clone-16/clone.pt --envs 512 --iterations 600 --minibatch 2048 --critic-warmup 60 --lr 1e-5 --kl-coef 2.0 --start-fraction 0.9 --eval-games 256 --eval-steps 20000 --eval-every 50 --out runs/snake-gpu/ppo-16`,
about 100 minutes. Use `--minibatch 2048` on 16x16: at 8192 the activations exceed the 12 GB of
video memory and an iteration stalls for minutes (measured: 2048 -> 8 s per iteration at a 4.1 GB
peak, 4096 -> 44 s at 6.6 GB). The baseline of that run is the clone's 76% on the batched
evaluation.

Re-measurement of the 16x16 clone (`runs/snake-gpu/clone-16/clone.pt`, last checkpoint) on 256 fresh
games, batched evaluation, greedy and unmasked, seed 123, step cap 65,536: 191 wins (74.6%, about
69-80% at 95% confidence), mean occupancy 77.5%. Failures: 62 starvation (the policy loops without
food), 3 collisions. Consistent with the 70% on the 20 tuning seeds. Next: keep the best checkpoint
by validation during cloning and decay the learning rate at the end; PPO only if that is not enough.

Second 16x16 clone (`runs/snake-gpu/clone-16b`, seed 1, 12,000 iterations, learning rate 1e-3 held for 70% of the
run then decayed linearly to 2e-5, 33 minutes): action accuracy 100.00% at the end. Re-measured on 256 fresh
games with the same protocol as the first clone (batched evaluation, seed 123, greedy, no mask, cap 65,536
steps): 252 wins (98.4%), mean occupancy 98.7%, mean 14,860 steps per game; the 4 failures are collisions and
there is no looping. On the 20 tuning seeds of the reference `SnakeEnv`: 20 of 20 wins. The first clone had 74.6%
on the same protocol, so the decay of the learning rate is what moved it. Caveat: the periodic evaluation of
this run used a step cap of 12,000, below the length of a winning game, so it reported 0% wins throughout and
the "best" checkpoint was chosen on occupancy only (it was the last one). The official 100-game acceptance on the
reference environment (seeds 3000000..3000099) is not yet run: the single-game CPU evaluation takes hours, a
batched reference harness is needed first.

Acceptance (2026-10-17): `runs/snake-gpu/clone-16b/best.pt` on the reference `SnakeEnv` (observation mode
`egocentric_cycle`), 16x16, 100 games on the frozen validation seeds 3000000..3000099, greedy, no action mask, no
teacher, step cap 65,536: **100 wins out of 100**, mean 15,156 steps per game, status `verified` (threshold: 90 of 100).
The validation seeds were used once, after the checkpoint was fixed; all tuning used other seeds. The model is a
network pretrained by imitating the Hamiltonian-cycle teacher and sees a cycle-order plane in its input; it was not
trained from scratch and PPO fine-tuning was not needed. Harness: `scripts/snake_gpu_verify.py`
(`evaluate_reference_batched`, 100 reference games stepped in lockstep with batched GPU inference, 100 s).
