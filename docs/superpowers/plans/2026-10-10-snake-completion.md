# Snake Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add completion-oriented Snake training and validate a learned preset on at least 16x16.

**Architecture:** Preserve the existing lab and process controls. Extend observations and checkpoint metadata, share lazy frame emission, and add Double DQN through the existing trainer. Separate reproducible performance validation from software verification.

**Tech Stack:** Python, Gymnasium, NumPy, PyTorch CPU, vanilla JavaScript.

**Spec:** `docs/superpowers/specs/2026-10-10-snake-completion-design.md`

## Global Constraints

- UI and communication in Russian; identifiers and documentation in English.
- Preserve existing working-tree changes and safely load version-1 models.
- Full occupancy on at least 16x16 is the requested learned result.
- No model is advertised as verified without independent game outcomes.
- Execute inline, as requested by the user; research/review may be delegated.

## Review Focus

- Switching live observation modes must rebuild the agent policy and observations together.
- A terminal transition must never bootstrap or double-pay the win bonus.
- Fast mode must avoid snapshots before the worker's transmission check.
- Best-policy playback must not substitute stale optimizer/replay state on resume.
- Held-out validation must not be reused for training or model selection.

### Task 1: Observations, rewards and evaluation

Files: `src/rl_fun/snake/{env,config,policy}.py`, `tests/snake/test_completion.py`.

Interfaces: `SnakeEnv(observation_mode='local', win_reward=0)`; `Policy(size=16, observation_mode='local')`; `evaluate(...)` retains score/reward/frame and adds occupancy, win_rate, wins, episodes, reasons, outcomes. `evaluation_rank(result)` returns the completion ranking tuple.

- [x] Add and run failing tests for ordered map, terminal win reward and evaluation outcomes.
- [x] Implement observation validation, 100000-step budget, win reward and metrics.
- [x] Run completion/environment tests and retain default legacy behavior.

### Task 2: Algorithms, checkpoints and lazy frames

Files: `src/rl_fun/snake/{dqn,evolution,checkpoint,worker,frames}.py`, `tests/snake/test_completion.py`.

Interfaces: `next_q_values(online, target, double)` selects online actions for Double DQN; `emit_frame(callback, env, **metadata)` constructs a snapshot only when requested. Version-2 checkpoint stores dynamic architecture and best_model separately from latest model.

- [x] Add and run failing tests for Double DQN, no per-step snapshots, legacy/v2 loads and continuation mismatch.
- [x] Implement shared algorithm updates and dynamic policy creation; keep existing DQN behavior.
- [x] Implement best-policy ranking and frame gate, including evolution and independent evaluation.
- [x] Run Snake tests, including exact mid-episode continuation and process controls.

### Task 3: Lab integration and preset

Files: `src/rl_fun/snake/{manager,api,presets}.py`, `src/rl_fun/lab/web/snake/{screen,chart}.js`, API/manager tests.

Interfaces: catalog adds observation modes and named presets; model records expose verification metadata. Playback uses best weights, continuation uses latest weights. Preset is Double DQN/board/16x16 with animation disabled.

- [x] Add failing integration tests for board-policy playback, live replacement and preset application.
- [x] Update catalog, resume validation and UI controls/metrics.
- [x] Verify old saved models, algorithm switching and training lifecycle.

### Task 4: Reproducible benchmark and empirical results

Files: `scripts/snake_check.py`, benchmark helpers/tests, README and execution ledger.

Interfaces: command line supports bounded training chunks, checkpoint continuation, fixed tuning seeds and independent 100-game validation; writes per-game outcomes and software metadata.

- [x] Test benchmark seed separation and verified-status threshold using actual result structures.
- [x] Implement and smoke-run the benchmark, then measure training throughput and learning.
- [ ] Train/tune on 16x16; report actual held-out results and continue evidence-based tuning.
- [x] Run the full test suite and lint; request one fresh read-only review and fix material findings.
- [x] Restart the local demo to expose the implemented controls and saved models.

## Execution ledger

Ruling: work in the current feature branch and working copy — the existing demo and uncommitted Snake work are here; a clean worktree would omit them.

Ruling: no automatic product-code commits — existing unrelated changes share these files; keep changes reviewable in place.

Ruling: the user's explicit instruction to implement is the execution handoff; do not repeat the earlier approval loop.

Demonstration-based initialization is a pending user preference and is not silently included.

Task 1: complete — ordered board inputs and terminal rewards passed environment checks.
Task 2: complete — Double DQN and legacy/v2 model round trips passed; lazy gate skips disabled and discarded frames.
Task 3: complete — browser verified preset application and an actual two-episode Double DQN run; no console errors.
Task 4: empirical acceptance pending — the first two runs measured 0/100 wins, improving mean food score from 6.45 to 21.51. Further bounded training is in progress. No successful completion claim.

Verification before review: full suite 1239 passed, 1 skipped; touched Python code passed Ruff. Full-repository Ruff also reports 38 existing errors outside these changes.

Fresh review identified replay validation, premature repeated held-out evaluation, and snapshots discarded at high FPS. Added regression tests, fixed target/replay validation and lazy should_emit gating, and separated tuning from explicit frozen-model validation. Final validation seeds now start at 3000000; the earlier exploratory seed set is never used as final acceptance.

Ruling: retain one version-2 checkpoint with latest model/optimizer/replay and a separate best_model entry — playback uses the best entry, continuation uses the latest entry, avoiding duplicated large replay files.

Ruling: completion benchmarking accepts board observation only and records the original training seed across resume — this keeps seed-range claims truthful.

Ruling: a won playback board remains visible until restart — automatic retry remains available for losing agents.

Final software verification: 1244 passed, 1 skipped; all changed Python files passed Ruff and diff whitespace checks. Browser preset application and training start/completion passed with no console errors.

Final bounded experiment: training continued to episode 4694, with 0/20 wins and 10.3% mean occupancy on tuning seeds. Frozen best-policy validation on the fresh 3000000..3000099 seeds produced 0/100 wins and 9.6% mean occupancy. Artifacts: `runs/lab/snake/af7562c7b46a4278967627854baae8db/validation.json`; preceding training: `runs/lab/snake/f0b75c2bde2a4941845ac68ab655e035/model.pt`. Performance acceptance remains unmet. All command-line training jobs ended; only the local demo server remains running.

Additional parameter search: three independent 180-second continuation budgets from f0b75c2bde2a4941845ac68ab655e035, varying one parameter at a time. Control: 0/20 tuning wins, 10.66% mean occupancy; epsilon_decay=10000: 0/20, 10.25%; death_penalty=50: 0/20, 10.25%. Reports: runs/lab/snake/parameter-search.json. Budgets are wall-clock, not matched steps, so these are exploratory comparisons. No held-out validation was used for selection. No verified completion preset was found; acceptance remains pending. The running demo exposes DQN, Double DQN and evolution; PPO is not implemented for Snake.

Scope extension explicitly requested by the user: add Policy Gradient and TD3 beyond Double DQN. Implemented categorical REINFORCE (full-episode Monte Carlo), synchronous n-step A2C, PPO (GAE, clipped surrogate, repeated minibatches), and TD3 with an explicitly experimental continuous-score argmax adapter. All use the existing actor checkpoint/playback contract and episode-based worker budget. Added algorithm dispatch and controls, entropy/noise metrics, CLI --algorithm, critic/optimizer/rollout validation, and exact PPO optimization-cursor resume.

Fresh review of on-policy training found missing control checks inside PPO minibatches and unchecked Adam moments. Added failing regressions, fixed cooperative interruption with a saved epoch/minibatch cursor, and validate finite optimizer state and moment shapes. TD3 review found time-limit bootstrap suppression; replay now stores true termination separately from episode end. An additional saturated-score regression ensures TD3 greedy playback uses the identical tanh adapter to training.

Browser verification: PPO completed three real episodes and saved model 73db7c09; TD3 completed three real episodes with actual actor updates and saved model 62c5bbf3. Fixed duplicate hidden form inputs discovered during browser TD3 submission. Saved TD3 playback loads and steps successfully, then was paused; console error list is empty. CLI PPO benchmark completed 176 episodes in a 10-second training budget, saved 6746d1d29d4f4ec3bad14f683470b5bf, and measured 0/20 tuning wins. These are execution checks, not a completion preset.
Final verification after TD3 playback fix: full suite 1268 passed, 1 skipped, 11 existing Gymnasium warnings. Changed Python files passed Ruff; git diff --check passed. Running demo PID 23844 exposes dqn, double_dqn, reinforce, a2c, ppo, td3 and evolution. No command-line training remains running.

Adaptive parameter search completed: 14 real experiments, 1,871,056 new environment steps on 16x16 board observations. Round 1 compared six algorithms with 100,000-step targets and 180-second training caps. Round 2 tested A2C entropy/value scaling, reward scaling, PPO horizon/reward variants and two Double DQN continuations. Round 3 continued the strongest new PPO and Double DQN for up to 400,000 further steps, capped at 360 training seconds. PPO reached 10.0% mean tuning occupancy after 400,000 additional steps; Double DQN reached 9.8% after 297,804 additional steps. Every new run had 0/20 wins. The older 4c310522 candidate remains strongest at 10.7%; its larger prior budget prevents a matched-budget comparison with these new runs. Completion acceptance remains unmet.

Artifacts: runs/lab/snake/search-round1.json, search-round2.json, search-round3.json, search-summary.json and reusable search-selected.json. No held-out seeds used for new tuning. Added bounded-step runner, latest-vs-best final selection, actual work/pending-rollout reporting and a measured experimental preset that loads matching saved weights. Fixed early-stop saved-policy mismatch; review also caught preset selection limited to 100 models, now it considers all saved models. Exact resume retains unfinished on-policy rollout. Training time caps exclude final evaluation.

Verification: all 98 Snake tests passed before preset integration; subsequent search/API tests 12 passed and manager/completion tests 25 passed. Touched Python files pass Ruff. Browser confirmed measured candidate selection, full map, actual parameters, correct selected model 4c310522, resume enabled, no console errors. Demo restarted as PID 2720; all search jobs ended. No winning preset was claimed.
