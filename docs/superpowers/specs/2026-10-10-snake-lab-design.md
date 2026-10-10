# Snake Lab design

Date: 2026-10-10
Status: approved and implemented

## Intent and scope

Extend the existing local ML laboratory with a Snake sandbox. The user can play,
train a neural agent using reinforcement learning or evolution, and compare human
and agent play side by side. This is a local addition to the existing Python/web
application, using installed NumPy, Gymnasium and PyTorch dependencies.

The first version includes real training, observable learning metrics, model
save/load and continuation. UI text is Russian. Code and identifiers are English.
The racing laboratory remains available. Existing uncommitted changes must be
preserved and accommodated.

## Game rules and environment

- Default board: 16 by 16 cells, configurable from 8 to 32.
- Initial snake: three cells, facing right; food occupies an empty cell.
- Walls and self-collision end the episode. Moving into the tail cell is valid
  when the tail moves away on that step. Filling the board wins the episode.
- Manual input uses arrows or WASD; reverse direction is rejected. Input is
  queued so two rapid turns cannot reverse the snake within a single tick.
- A configurable starvation budget, default 100 times board area, truncates an
  episode without food; eating resets this budget.
- Food placement uses an environment-owned seeded random generator. A reset
  with the same seed reproduces the initial state and food sequence.
- Agent actions are relative: straight, turn left, turn right. Manual absolute
  direction input is translated into these same actions on the server.

The Python environment is authoritative for manual play, training and evaluation.
It exposes Gymnasium reset/step and JSON-safe snapshots for rendering. It is also
registered in the workbench environment registry for independent experiments.

The first observation is an 11-element float vector: danger straight/left/right,
four direction indicators, and four indicators locating food relative to the
head. Danger uses the same collision rules as step. Reward is +10 for food,
-10 for collision, and a small step penalty (-0.01); completion gives the food
reward and terminates successfully. Starvation truncates without a death penalty.
Scores always report food eaten separately from cumulative training reward.

## Training

Both algorithms use a small MLP with an 11-element input, two hidden layers of
64 units and three output values. Both share observation/action conventions.

### DQN

Use replay memory, minibatch TD updates, a periodically synchronized target
network, epsilon-greedy exploration and terminal-aware bootstrap targets. Bound
replay capacity and perform updates at a configurable interval. Use an Adam
optimizer and CPU execution by default. User controls include episode budget,
learning rate, discount, batch size and exploration schedule. Expose completed
episodes, reward, score, rolling mean score, epsilon and loss.

Training ends on the configured episode budget or explicit stop. Pausing retains
the environment, model, replay buffer and optimizer state. Greedy evaluation
uses independent environments and does not consume training random state.

### Evolution

Evaluate a population of MLP parameter vectors across shared seeded episodes.
Rank by average return, retain elites, sample parents from the better-performing
portion and mutate their weights with Gaussian noise. Keep the best evaluated
policy available for playback. User controls include generations, population,
elite fraction, mutation scale and evaluation episodes. Expose generation,
mean/best return and mean/best food score. Bound population and episode budgets
to keep memory and stop latency reasonable.

Each evaluation seed set is shared within a generation; a separate fixed seed
set reports comparable evaluation progress. Do not equate reward with score.

## Interface and play modes

Add a Snake entry to the existing navigation and a dedicated screen using the
current theme, components and router. Keep Snake rendering and controls in
separate modules rather than extending racing-specific view logic.

- Manual: one board, score and controls for start, pause and restart.
- Agent: one board showing the selected policy, with episode and speed controls.
- Split: human board on the left, agent board on the right. Both start with the
  same seed, board size and tick rate. Each progresses independently; death on
  one side does not stop the other. Replay starts both sides from the same seed.
- Training panel: algorithm selection, parameters, start/pause/resume/stop,
  progress and score/reward charts, and save/load controls.

On narrow screens the boards stack, human first. Keyboard events affect only
the focused game and do not intercept form inputs. Losing focus pauses manual
play. Display the actual selected model and distinguish an untrained network
from a trained one. Before training, manual play works and agent modes can run
an explicitly labeled untrained network; no scripted policy is labeled neural.

Playback speed affects game presentation, not the training budget. Rendering
and training are decoupled. Limit chart history and snapshot frequency.

## Server and runtime architecture

Introduce a Snake package containing environment, policy, DQN, evolution and
checkpoint functionality. Provide a Snake-specific manager/worker and API under
`/api/snake`, owned by the existing local server alongside the racing manager.
Do not route Snake frames through racing-only configuration or protocols.

Run training in a separate process to keep the HTTP server responsive. Publish
bounded snapshots and metrics over the existing SSE transport patterns. Expose
create/status/control operations for training and game sessions. Manual ticks
and agent ticks use the same environment implementation. Only one Snake
training job is active at a time in the first version; starting another requires
stopping the current job. Finished jobs remain inspectable and their policy can
be loaded into playback.

Validate all numeric fields, actions and lifecycle transitions. Retain existing
loopback, Host validation and request-size protections. Reject incompatible
checkpoints with a readable error. Server shutdown closes sessions and joins
or terminates owned training workers; crashed workers surface a failed state.

## Persistence and continuation

Store Snake runs separately beneath the configured runs directory. Checkpoints
contain a format version, environment settings, algorithm, model architecture,
weights, training counters and algorithm-specific continuation state. DQN
continuation includes optimizer, target network, replay state and random state;
evolution includes population, best evaluated policy and random state.

Use structured metadata and tensor/array data with safe loading; never execute
arbitrary pickled objects. Save via atomic replacement. API model identifiers
resolve within the managed runs directory and cannot select arbitrary paths.
Saving/loading is exposed through a local model list rather than requiring a
general file-upload facility. Loading for playback is separate from resuming
training. Persist metrics so completed runs can be reopened after a restart.

## Verification and acceptance

- Environment tests cover seeded resets, food, growth, collisions, tail moves,
  starvation, full-board completion, action mapping and observation consistency.
- DQN tests verify terminal bootstrap masking and real parameter updates.
- Evolution tests verify elite retention, mutation and comparable seed sets.
- Round-trip checkpoint tests verify predictions and continuation state.
- Server tests cover validated controls, lifecycle, worker failures, shutdown,
  isolated game sessions and checkpoint path confinement.
- UI checks cover navigation, keyboard focus, pause/restart, all play modes,
  model selection and training metrics. Run relevant racing regression tests.
- Smoke-train each algorithm with small budgets, then save/load and play its
  policy. A short smoke run verifies execution, not achieved playing strength.

The feature is accepted when a user can launch the existing lab, play Snake,
train either algorithm, inspect genuine metrics, save/load a model, continue
training, and run human-left/agent-right play without freezing the interface.
Learning strength is measured through seeded evaluation and is not guaranteed
by a fixed short training duration.

## Deferred scope

Custom code editing in the browser, additional RL algorithms, tournaments,
remote hosting, GPU tuning and simultaneous multiple Snake training jobs are
outside the first version. The Python environment and learner interfaces remain
available for adding custom algorithms in code.
