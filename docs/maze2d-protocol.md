# Maze2D U-Maze: the paper evaluation protocol

This example targets the paper's **Inverter results on `maze2d-umaze-v1`**.
It includes four archived Inverter checkpoints and their paired forward models.
The original baseline implementations and other maze tasks are outside this
first example.

## Environment and inputs

Use the [pinned legacy environment](maze2d-environment.md). The benchmark is
`maze2d-umaze-v1`, with Gym, D4RL 1.1, `mujoco-py`, and native MuJoCo 2.1.0.
The `maze2d-eval-umaze-v1` task and newer Gymnasium/Minari environments are
different reproduction targets.

The point mass observes raw `[x, y, vx, vy]`. The Inverter receives this state
and the fixed goal state `[1, 1, 0, 0]`, and predicts 128 actions with two
components each. Observations are not standardized or translated. Keep the
original maze geometry and leave `MAZE2D_BODY_POS` and `MAZE2D_CENTERED` unset.
The original body offset is `[1.2, 1.2, 0]`; moving it to `[1, 1, 0]` changes the
maze relative to the observed coordinates.

The deployment controller needs only an Inverter checkpoint. It uses no
waypoint planner, occupancy grid, forward-model rollout, or offline training
data at evaluation time.

## Episodes and action execution

For **each checkpoint and each execution length K**:

1. Run 100 episodes. For episode `i = 0, ..., 99`, call
   `env.seed(777 + i)` followed by `env.reset()`.
2. Keep the benchmark's fixed goal `(1, 1)`; do not sample a new target.
3. Generate 128 actions, execute the first `K`, observe the new state, and
   generate a new sequence. Actions are float32 and clipped to `[-1, 1]`.
4. Execute 300 environment steps per episode. Continue using Inverter actions
   after reaching the goal. Do not terminate on first arrival or substitute a
   proportional–derivative holding controller.
5. Sum the environment's sparse rewards over all 300 steps.

The initial state differs across episodes but is shared across checkpoints and
K settings. The first three cases are useful compatibility checks:

| Reset seed | Initial x | Initial y | Initial vx | Initial vy |
| --- | ---: | ---: | ---: | ---: |
| 777 | 0.9765633345 | 1.0200140476 | −0.1583636701 | 0.0632457584 |
| 778 | 0.9017087221 | 0.9498471022 | −0.0097955102 | −0.0030397940 |
| 779 | 2.9001364708 | 3.0082192421 | 0.0160125438 | 0.0318456516 |

## Scores and uncertainty

Report `100 * env.get_normalized_score(mean_episode_return)`. For this task,

```text
normalized_score = 100 * (mean_episode_return - 23.85) / (161.86 - 23.85)
```

A score above 100 is valid. It is a normalized return, not a success percentage.
D4RL's reward uses distance at most 0.5 from the goal. The separate navigation
success diagnostic uses distance below 0.45 at any point in an episode.

Calculate a score for each of the four model checkpoints, then report the mean
and **population standard deviation across those four scores** (`ddof=0`). The
standard deviation across individual episodes is a different quantity.

| Actions executed before replanning, K | Calls per 300-step episode | Paper score | SD across models |
| ---: | ---: | ---: | ---: |
| 16 | 19 | 164.25 | 0.54 |
| 32 | 10 | 164.77 | 0.34 |
| 64 | 5 | 165.19 | 0.80 |
| 128 | 3 | 161.64 | 2.17 |

The final chunk is truncated at step 300. At K=128, calls occur at steps 0, 128,
and 256; the last call supplies 44 executed actions.

The frozen reference originates from the research archive's
`maze2d/timing/timing_umaze_4seeds_evalseed777_20260426_083101.csv`.
Its SHA-256 is
`271ea4d65e838ae49f4373ff163f596df95486490650d79b6d15e545bd76d152`.
The example's reference results retain the per-model values and source
provenance. Future runs write separate output files.

## What the four model labels mean

| Checkpoint family | Forward-model data-sampling seed | Inverter data-sampling seed |
| --- | ---: | ---: |
| `default` | 42 | 123 |
| `fmseed123` | 123 | 123 |
| `fmseed456` | 456 | 123 |
| `fmseed789` | 789 | 123 |

Each Inverter was trained through its own paired forward model. All four
archived Inverter configurations use data-sampling seed 123. These labels
identify checkpoint families; they are not four different Inverter seed values.
Use the supplied validation-loss-selected `best_model.pt` weights, packaged as
`inverter.pt` and `forward.pt`. The separate historical `best_d4rl_model.pt`
checkpoints do not define this comparison.

## Training details that matter

The saved configurations are the reference for the training recipe:

- Training uses raw float32 observations and actions from the D4RL offline data.
- The forward model predicts 16-step chunks. The Inverter emits 128 actions;
  training differentiates through eight successive frozen forward-model chunks.
- The Inverter's dense predicted goal-reward and occupancy-boundary penalties
  both have weight 5. The terminal-state loss weight is 0.
- The occupancy map is a 64×64 binary visit grid smoothed with Gaussian sigma
  1.5 cells. It is constructed from offline states.
- The archived Inverter runs use `--no_episode_aware`: they sample fixed-horizon
  windows in the concatenated offline arrays. Goal-change/timeout boundaries
  in this data do not necessarily represent physical resets.
- Training and validation windows are sampled independently with replacement
  from the same offline array. They are not a disjoint trajectory split.
- The original `--seed` flags control NumPy sampling; the original trainers do
  not completely seed PyTorch model initialization. Reproducing the recipe
  therefore does not guarantee identical weights. Use the archived checkpoints
  to evaluate the saved models.

These instructions describe the archived execution. In particular, the saved
flat-window sampling option differs from the paper's episode-aware prose.

## Runtime measurements

The paper's GPU timing campaign used an NVIDIA A40 and counted planning calls
with CUDA synchronization. It discarded ten warm-up calls and used inference
mode without gradients. The research evaluator additionally performed a
forward-model rollout for visualization; that cost was subtracted when
constructing the paper's inference-budget table.

The tutorial evaluator omits visualization and reports its own measured costs.
Use the same machine, warm-up and synchronization rules, and timing boundaries
when making a runtime comparison. CPU runs check behavior and the evaluation
protocol; they do not reproduce the reported A40 latency.

## Reproduction checklist

- [ ] The environment check passes for the pinned legacy stack and canonical reset states.
- [ ] The task is exactly `maze2d-umaze-v1` with fixed goal `(1, 1)`.
- [ ] The archived checkpoint hashes match the manifest.
- [ ] All four models use reset seeds 777–876 and 100 episodes each.
- [ ] Every episode has 300 steps, including time after first arrival.
- [ ] The selected K is explicit, with no PD holding controller or routing planner.
- [ ] Scores use D4RL normalization and SD across model scores.
- [ ] The saved output identifies the environment, checkpoint, K, device, and seeds.
