# Snake completion and reproducible training

Date: 2026-10-10
Status: software implemented; learned completion acceptance pending

## Approved intent

The user wants an agent to finish Snake by occupying every cell, fast training
with rendering disabled, another RL algorithm, and a preset backed by an actual
trained model. The approved approach is to improve observations, add Double DQN,
measure completion, and establish a verified completion preset for a board of
at least 16x16. The user's subsequent clarification makes 16x16 the minimum
acceptance size. Small-board experiments may inform tuning but do not satisfy
the requested result. All UI text is Russian; code and artifact identifiers
are English.

This extends the existing Snake lab. Preserve all existing uncommitted work.
No deterministic solver or scripted expert is presented as a learned policy.

## Current behavior and motivation

SnakeEnv already ends with reason `win` when no free cell remains. Initial length
is three, so a board of side N requires N*N-3 food items. The current environment
accepts sides 8 through 32. DQN and evolution use an 11-feature observation and
an 11 -> 64 -> 64 -> 3 MLP. These features do not describe the body's layout.

The animation toggle suppresses transmitted frames and FPS delays, but learners
still construct snapshots on every step. DQN evaluation is capped at 500 steps.
Evaluation reports food score and reward without a win rate. Existing saved
16x16 runs provide no evidence of full-board completion.

## Game objective and observations

Keep the existing collision, food, growth, relative-action and moving-tail rules.
The only successful terminal outcome is full occupancy. Expose occupancy as
snake length divided by board area, including the initial three cells. Food
score remains a separate quantity. Collision, starvation, repeated-state loops
and step-budget exhaustion are failures to complete, not wins.

Retain sizes 8 through 32 in the environment, configuration, API and UI. The
completion preset uses 16x16 and must eat 253 food items to occupy all 256
cells. Smaller supported boards may be used for intermediate experiments;
their results cannot qualify the completion preset. No size-4 extension is
needed for this change.

Support two explicit observation modes:

- `local`: the existing 11 features, with the existing 64/64 hidden layers.
- `board`: the 11 features followed by two flattened N by N planes. The first
  encodes body order: empty=0, tail=1/length, subsequent segments increase to
  head=1. The second encodes food=1 and all other cells=0. A completed board
  has an all-zero food plane. Flatten in row-major order using the same x/y
  convention as the environment. Use an MLP with hidden layers 128 and 128.

The body-order plane supplies both occupancy and which occupied cells vacate
first. Input dimension is 11+2*N*N; observation-space bounds and actual arrays
must match. Board size and observation mode are structural model settings.

Existing configurations without an observation field retain `local`. New
completion presets explicitly select `board`. Manual games need no model;
agent playback obtains its observation mode from the selected checkpoint.

## Rewards and algorithms

Retain DQN and evolution and add the algorithm identifier `double_dqn`.
Double DQN uses online-network argmax selection for the next action and the
target network to evaluate that action. Reuse the DQN replay, optimizer,
exploration, pause, save and continuation mechanisms. Ordinary DQN keeps its
current target calculation. No new dependency is required.

Add a configurable nonnegative `win_reward`, paid once on the terminal food
step in addition to the food reward. Its legacy default is zero. Existing
food, death, ordinary-step and idle reward controls remain available. Do not
silently add an expert, route-following policy or action override to training.

## Fast mode and live visualization

Keep the existing animation toggle, relabel or explain it as fast training
when disabled, and default the completion preset to animation disabled.
The animation FPS controls simulation pacing only while animation is enabled.

Introduce a shared lazy frame-emission contract used by DQN, evolution and
evaluation. Process pending control messages before deciding whether a frame
is needed. Only construct a snapshot when visualization is enabled and a
frame is due. Disabled animation must perform no per-step snapshot creation,
frame transmission or FPS sleep. Switching animation on during an active run
must resume frames without restarting training. Metric emission, checkpoints
and pause/stop responsiveness continue in either mode. A final summary snapshot
is allowed; avoiding per-step work is the requirement.

## Evaluation and model selection

Use the configured evaluation step budget rather than an implicit 500-step cap.
Increase the accepted max_steps upper bound to 100000 to permit long completion
episodes, with control polling retained on every step. A repeated full state
with unchanged food may still end a deterministic evaluation as `loop`.

Each completed evaluation records episode count, win count, win rate, mean and
maximum occupancy, mean score, mean reward, and counts of terminal reasons.
An external step-budget exhaustion receives an explicit `max_steps` reason.
Use these definitions for both neural learning algorithms and evolution.
Cancellation returns no partial result advertised as a completed evaluation.

Select the best evaluated model lexicographically by win rate, mean occupancy,
then mean food score. Preserve a separate latest training checkpoint for exact
continuation: the best evaluation policy need not be the latest optimizer state.
Existing playback/checkpoint consumers continue to work for legacy models.
Training charts and summaries display occupancy percentage and win rate, while
retaining the existing food/reward metrics. Final successful playback explicitly
shows full-board victory.

## Checkpoint and configuration compatibility

Continue safely loading version-1 tensor checkpoints with their exact legacy
architecture and missing fields filled with legacy defaults. New checkpoints
use version 2 and include observation mode, board size, full architecture and
algorithm metadata. Validate those fields against weights and replay shapes.

Playback, evaluation and trainers construct the matching policy/environment.
Continuation rejects incompatible size, observation mode or algorithm with a
readable error. DQN and Double DQN are distinct continuation algorithms even
though they share implementation. Never reinterpret an old replay buffer as
board observations or silently resize trained weights.

## Presets and empirical acceptance

Expose a named completion-training preset in the UI and a reproducible command
line benchmark. Applying a preset updates the visible configuration, board size
and observation mode together. It is not called verified until evidence exists.

The initial search configuration is a candidate, not an assertion of ideal
parameters: Double DQN, board observation, size 16, learning rate 0.0003,
gamma 0.999, batch size 64, replay capacity 50000, warmup 1000,
train interval 4, target interval 500, epsilon 1.0 down to 0.02 over
60000 episodes, episode budget 100000, max_steps 100000, food reward 10,
death penalty 10, win reward 100, ordinary-step penalty 0.001,
idle_steps 0, animation disabled. All settings are explicit in saved artifacts.
These are starting search settings, not a verified winning recipe. Measure
throughput before a full run and use checkpointed training in bounded chunks.

Tune on a fixed 20-seed evaluation set, then freeze the chosen configuration
and model before evaluating 100 separate held-out seeds. Record training,
tuning and validation seed sets and verify that they are disjoint. Validation
uses greedy actions without exploration and the same game/reward rules. Both
tuning and final acceptance evaluate the chosen model on 16x16. If a larger
board is offered as the completion preset, validate that exact size as well.

Save the complete configuration, training seed, model checkpoint, per-episode
validation outcomes, aggregate metrics, software/Git version, dirty-worktree
status and elapsed time under the existing runs directory. Show the measured
win rate and board size beside a verified preset. Avoid a machine-dependent
absolute path or a hard-coded run ID in the UI; discover benchmark artifacts
from the configured runs directory.

A model qualifies as demonstrating the requested learned completion only after
an actual held-out full-board win on a board of at least 16x16. A preset
qualifies as a reliable completion preset only after at least 90 wins out of
100 held-out games at its advertised size, which must be at least 16x16.
Record results even if
the threshold is missed and keep the preset marked experimental. Passing code
tests or one lucky win is not evidence of a reliable preset. Larger boards
require their own training and validation; no generalization claim is implied.

Implementation acceptance and learned-performance acceptance are separate.
If the initial training budget does not produce the requested performance,
report the actual results and continue targeted tuning rather than inventing a
successful model. A further training budget can be discussed using measured
runtime and learning curves; a long-running job is not silently left behind.

## Verification

- Environment: full-board terminal reward once, occupancy, size-16 reset,
  ordered-body/food planes, observation shape, tail movement, seeded food.
- Algorithms: Double DQN action-selection/value split and terminal masking;
  both algorithms perform real parameter updates with each observation mode.
- Compatibility: version-1 playback and continuation; version-2 round trips;
  structural mismatch rejection; separate best/latest policies.
- Fast mode: count snapshots and mock sleeps to prove absence of per-step work
  while disabled, including evaluation and evolution, and test live re-enabling.
- Evaluation: known winners and failures, full budgets beyond 500 steps,
  reason counts, cancellation, occupancy calculation and model ranking.
- Integration/UI: validated algorithm/observation/preset fields, Russian labels,
  playback of old and new models, training controls and completion summaries.
- Run the relevant Snake and laboratory tests and lint. Run a small training
  smoke check, then the real benchmark; preserve and disclose actual outcomes.

## Scope boundaries

This change does not add PPO, recurrent policies, convolutional networks,
curriculum across board sizes, parallel training jobs, GPU support or a
deterministic completion agent. Double DQN supplies the requested additional
RL option while keeping the first completion experiment reproducible.

## User-requested algorithm extension

The later explicit request for Policy Gradient, TD3 and other algorithms supersedes the initial exclusion of PPO. Snake now supports REINFORCE, A2C and PPO with categorical actions and an experimental TD3 action adapter (three continuous preference scores mapped to a turn by argmax). Algorithm availability is separate from learned-completion acceptance; no successful full-board preset has been established.
