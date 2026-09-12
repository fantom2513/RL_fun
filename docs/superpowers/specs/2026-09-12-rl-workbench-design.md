# RL Workbench Foundation Design

**Date:** 2026-09-12  
**Status:** Approved in conversation; awaiting review of the written specification

## 1. Purpose

RL Fun is a local reinforcement-learning workbench for learning algorithms, running reproducible experiments, and later building custom interactive environments. It should support two complementary workflows:

- Jupyter notebooks for explanations, formulas, plots, and hands-on modification;
- Python scripts for repeatable training, evaluation, comparison, and visualization.

The first milestone establishes the development and experiment runtime, then validates it with a small multi-armed-bandit experiment. It does not implement Snake, a general game engine, or advanced deep-RL algorithms.

## 2. Goals

- Use Gymnasium as the public contract for RL environments.
- Keep simple algorithms short, explicit, and readable.
- Run one experiment or a series of independent seeds through the same API.
- Run CPU experiments concurrently without mixing state or artifacts.
- Save enough metadata and outputs to reproduce and compare runs.
- Allow the same implementation to be used from notebooks and scripts.
- Keep visualization independent from simulation and training.
- Leave narrow extension points for a Pygame arena, neural-network inspection, Stable-Baselines3, browser rendering, and multi-agent environments.

## 3. Non-goals for the first milestone

- Distributed or cloud runners.
- Remote workers, job queues, or cluster recovery.
- Hyperparameter-sweep infrastructure beyond explicitly supplied run variants.
- MLflow, hosted experiment tracking, or cloud artifact storage.
- A universal Trainer hierarchy or general plugin framework.
- Course-style autograding of learner-written algorithms.
- Pygame Arena, Neural Inspector, browser UI, Snake, or physics environments.
- DQN, PPO, actor-critic, or other advanced neural algorithms.

## 4. Design principles

### 4.1 Lightweight core

The core owns experiment execution, reproducibility, metrics, and artifacts. Individual algorithms own their learning loops. A shared abstraction is added only after at least two real uses demonstrate the same need.

### 4.2 Notebooks are clients

Notebooks import reusable code from `src/rl_fun`; they do not maintain separate copies of algorithms or runner logic. A notebook may contain deliberately incomplete exploratory code, but canonical ready-to-run implementations remain in the package.

### 4.3 Short algorithms

Basic algorithms use small functions or compact classes in one file per concept. Their updates remain visible rather than being hidden behind a generic Trainer, callback tree, or dependency-injection framework.

### 4.4 Separate concerns called “environment”

The Python development/runtime environment is distinct from a Gymnasium task environment. Within future task environments, state transitions and emitted domain events should remain separate from experiment-specific reward shaping when the problem permits it.

### 4.5 Rendering is observational

Training must not depend on a renderer. Future renderers consume bounded snapshots at their own display rate; they never define the simulator clock or block the learning loop.

## 5. High-level architecture

```text
Jupyter notebooks       Scripts / CLI       Arena / Inspector (later)
        |                    |                         |
        +--------------------+-------------------------+
                             |
                     Experiment Runner
                  single run / multi-seed plan
                             |
            +----------------+----------------+
            |                                 |
     Gymnasium environment              Short algorithm
            |                          optional PyTorch model
            +----------------+----------------+
                             |
                          RunResult
                             |
            +----------------+----------------+
            |                |                |
       metrics sink     artifact store   snapshot sink (later)
            |                |                |
       plots/TensorBoard   runs/          Pygame/browser (later)
```

## 6. Project layout

```text
RL_fun/
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
├── notebooks/
│   ├── 01_bandits/
│   ├── 02_tabular/
│   ├── 03_dqn/
│   └── 04_policy_gradient/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── compare.py
│   └── arena.py                 # introduced with the visualization milestone
├── src/rl_fun/
│   ├── algorithms/
│   ├── environments/
│   ├── experiments/
│   ├── models/
│   ├── tracking/
│   ├── visualization/
│   └── baselines/
├── tests/
└── runs/                        # generated and ignored by Git
```

Empty packages are not created merely to match this tree. Directories appear when their first real module is implemented.

## 7. Toolchain and dependencies

The project targets Python 3.12 on Windows 11 and uses `uv` with `pyproject.toml` and a committed `uv.lock` file.

Planned dependency groups:

- core: Gymnasium and NumPy;
- notebook: Jupyter and Matplotlib;
- learning: PyTorch with the appropriate CUDA wheel source;
- baselines: Stable-Baselines3 and TensorBoard;
- visualization: Pygame CE;
- development: pytest and Ruff.

Only the groups required by a milestone are installed. The first bandit milestone does not require PyTorch, Stable-Baselines3, TensorBoard, or Pygame. Exact versions and the PyTorch CUDA index are resolved and verified during implementation; dependencies are not installed merely because they appear in this design.

## 8. Core contracts

### 8.1 Experiment configuration

The first milestone uses a typed dataclass for shared run settings: experiment name, base seed or explicit seeds, number of steps, parallel worker count, and output root. A concrete experiment may compose or extend this with its own algorithm and environment settings.

Configuration can be constructed directly in Python, which keeps notebooks simple. In the first milestone, scripts load JSON config files and allow a limited set of command-line overrides. JSON requires no additional parser dependency and is also used for the exact configuration snapshot stored with a run. The design does not introduce Hydra.

### 8.2 Algorithm entry point

Each basic algorithm exposes a small, direct entry point that receives its task/environment, typed settings, random generator, and metric sink, and returns a `RunResult`. The algorithm retains ownership of its loop and learned state.

The contract standardizes inputs and outputs, not the internal structure of every algorithm. Tabular and neural algorithms do not have to inherit from one universal trainer.

### 8.3 Metric sink

The minimal operation is conceptually `log(step, metrics)`, where metrics is a mapping of stable names to scalar values. Initial sinks are:

- in-memory collection for notebooks and tests;
- local JSONL or CSV output for durable runs.

TensorBoard becomes an adapter during the neural-learning milestone. Algorithm code does not import TensorBoard directly.

### 8.4 Run result

A result contains the final metrics summary, learned algorithm state when applicable, run status, seed, elapsed time, and artifact references. Large histories stay in metric files rather than being duplicated in the result object.

### 8.5 Snapshot sink

The first milestone documents but does not implement the live visualization path. Later, a bounded snapshot contract carries environment frames, policy outputs, and optionally sampled neural activations. Pygame is the first consumer; a browser renderer may consume the same protocol later.

## 9. Experiment data flow

1. A notebook or script constructs an experiment configuration.
2. The runner validates it before creating outputs or workers.
3. A run plan expands the base configuration into independent seeds.
4. Each run creates its own environment, algorithm state, random generator, metric sink, and artifact directory.
5. The algorithm executes its explicit learning loop and logs scalar events.
6. The runner stores metadata, metrics, the result summary, and a checkpoint when the algorithm supports one.
7. The parent process aggregates only completed summaries; it never shares mutable learning state between workers.
8. A notebook or comparison script reads stored results and calculates aggregate statistics across seeds.

## 10. Parallel execution

CPU runs may use separate processes. The design must be compatible with Windows `spawn` semantics: worker entry points are importable functions, process creation is guarded, and configurations/results are serializable.

Every worker writes to a unique run directory. This avoids locks and prevents metric files from being interleaved.

GPU jobs are not blindly launched in parallel. A later neural milestone may default to one GPU training job at a time while allowing CPU evaluation or rendering alongside it. Explicit concurrency can be enabled when measured memory requirements permit it.

## 11. Artifacts and run directories

Each run receives a stable unique identifier and a directory similar to:

```text
runs/<experiment>/<timestamp>-<run-id>/
├── config.json
├── metadata.json
├── metrics.jsonl
├── summary.json
├── checkpoint.*               # only when supported
├── error.txt                  # only on failure
└── media/                     # only when produced
```

Metadata records Python and package versions, operating system, CPU/GPU information, CUDA availability, seed, Git commit, and whether the working tree was dirty. Generated runs are excluded from Git.

Checkpoint writes use a temporary file followed by an atomic replacement. Checkpoint and resume behavior is implemented fully when the first stateful neural algorithm requires it; the foundation defines the artifact boundary without pretending every bandit needs a heavy checkpoint protocol.

## 12. Failure handling

- Invalid settings fail before worker creation.
- Failure of one seeded run is captured with its traceback and does not erase successful sibling runs.
- Batch summaries distinguish success, failure, cancellation, and incomplete work.
- `Ctrl+C` stops child processes cleanly and preserves already completed artifacts.
- Device selection is explicit. A requested unavailable GPU produces a clear error unless the configuration explicitly allows CPU fallback.
- Corrupt or incompatible checkpoints fail with actionable context rather than silently starting a fresh run.

## 13. Visualization architecture for later milestones

### 13.1 Comparative arena

The arena loads policies and steps multiple independent copies of the same environment in sync. For fair visual comparison, copies can share scenario parameters and derived seeds. The first comparison mode shows separate worlds side by side; shared-world interaction belongs to a future multi-agent design.

Controls include pause, resume, restart, speed, and single-step. Simulation speed is independent of display refresh rate. The arena can record one composite video.

### 13.2 Neural Inspector

PyTorch hooks sample selected tensors without modifying policy behavior. For small multilayer perceptrons, the inspector can display every neuron and connection:

- node brightness or fill represents activation;
- edge color represents positive or negative influence;
- edge thickness represents weight magnitude or current contribution;
- output nodes show action values or probabilities;
- the selected action is highlighted.

Full connection graphs are not used for large networks. Convolutional or large models receive aggregated layer summaries, feature maps, histograms, or other bounded views. Tensors are detached, sampled at a configured interval, and moved off the training device asynchronously where practical.

The initial Neural Inspector milestone targets inference and playback. Live training inspection is added only after its performance cost is measured.

## 14. Ready implementations and learning path

The workbench develops in increasing conceptual complexity:

1. random, greedy, epsilon-greedy, optimistic initialization, and UCB;
2. dynamic programming, Monte Carlo, TD(0), SARSA, and Q-learning;
3. function approximation and DQN;
4. policy gradients, actor-critic, and PPO.

Basic algorithms receive short in-project reference implementations. Stable-Baselines3 is introduced later as an external baseline for neural algorithms, not as the internal architecture of the workbench.

Educational notebooks contain explanation, formulas, experiments, and optional places for manual modification. They are not backed by a comprehensive learner autograder.

## 15. Testing strategy

Infrastructure tests cover:

- deterministic output for the same seed on a deterministic fixture;
- isolation of metrics and artifacts between parallel workers;
- configuration serialization and deserialization;
- checkpoint save/load when checkpoint support is introduced;
- clean reporting of partial batch failures;
- a short end-to-end smoke experiment;
- Gymnasium `check_env` for every custom environment.

Algorithm tests remain proportional to the algorithm. Canonical implementations receive focused correctness checks, but the project does not include dozens of course-style tests intended to grade every learner exercise.

## 16. First milestone

The first vertical slice contains:

- `uv` project setup and locked dependencies required for the slice;
- importable `rl_fun` package;
- typed experiment settings;
- single-run and multi-seed CPU execution;
- memory and local-file metric sinks;
- run directories, metadata, and summaries;
- a small stationary multi-armed-bandit task exposed as a Gymnasium environment;
- random, greedy, and epsilon-greedy reference implementations;
- a comparison script or function for aggregate results;
- one explanatory notebook using the same package code;
- focused infrastructure tests and one end-to-end smoke test.

Expected size is approximately 1,500–2,500 lines including tests and notebook content. The estimate is a scope guard, not a target to fill.

### Acceptance criteria

- A fresh checkout can be synchronized and run through `uv` using the committed lockfile.
- The same bandit experiment runs from both a script and the notebook.
- A user can run several seeds, optionally in parallel on CPU.
- Each seeded run has isolated metrics and metadata.
- The comparison output reports aggregate performance across seeds.
- Repeating the same configuration and seed reproduces the same bandit trajectory and result on the same software/hardware setup.
- The documented test command completes successfully.

## 17. Later milestones

1. Tabular RL and small Gymnasium environments.
2. Pygame comparative arena backed by the snapshot contract.
3. PyTorch foundations and DQN.
4. Stable-Baselines3 and TensorBoard adapters.
5. Neural Inspector for inference/playback.
6. A reusable custom-environment template and Snake.
7. Policy-gradient algorithms and live neural inspection.
8. Multi-agent or physics backends only when a concrete task requires them.

Each milestone gets its own small design adjustment and implementation plan. The workbench does not pre-build unused framework layers for later items.

## 18. References informing the design

- Gymnasium API and utilities: <https://gymnasium.farama.org/>
- uv project and lockfile model: <https://docs.astral.sh/uv/concepts/projects/layout/>
- PyTorch local installation: <https://pytorch.org/get-started/locally/>
- PyTorch TensorBoard integration: <https://docs.pytorch.org/docs/stable/tensorboard>
- Stable-Baselines3 Gymnasium integration: <https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html>
- PettingZoo AEC API for future multi-agent work: <https://pettingzoo.farama.org/main/api/aec/>
