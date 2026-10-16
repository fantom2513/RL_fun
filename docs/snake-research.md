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
