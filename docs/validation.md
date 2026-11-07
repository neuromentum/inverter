# Validation of the first U-Maze snapshot

Checked on 2026-11-07 using the existing Linux research environment. This record
separates executed checks from the longer training and setup instructions.

## Checks completed

- Installed the package in editable mode from outside the checkout, using an
  isolated virtual environment with access to the existing dependencies.
- Passed the environment check, including canonical geometry, goal, seed-777
  reset, native MuJoCo version, and one simulator transition.
- Repeated the check with an **unmodified copy of the pinned public D4RL 1.1
  wheel**. The tutorial does not require a private D4RL patch.
- Verified all eight archived forward-model/Inverter checkpoint hashes.
- Extracted the cached original HDF5 data and reproduced all five NumPy-array
  hashes.
- Ran each training/data entry point's command-line help.
- Completed one small CPU training epoch for the full forward-model architecture
  and one for the full Inverter through that newly trained forward model. Each
  used four training windows, four validation windows, and batch size two. Both
  saved valid checkpoints; the Inverter also ran one simulator diagnostic.
- Passed the evaluator's protocol checks for 300-step continuation, action
  clipping, chunk truncation, distinct reward/success radii, and rejection of
  unexpected early termination.

## Full archived-checkpoint evaluation

The CPU evaluation covered all four checkpoints and all four execution lengths:
**1,600 episodes and 480,000 simulator steps**. All reset states, episode lengths,
controller-call counts, and success outcomes matched the reference.

| K | Archived mean | CPU mean | Mean difference |
| ---: | ---: | ---: | ---: |
| 16 | 164.25259 | 164.25259 | 0.00000 |
| 32 | 164.77248 | 164.77429 | +0.00181 |
| 64 | 165.19455 | 165.19636 | +0.00181 |
| 128 | 161.63865 | 161.65676 | +0.01811 |

Eleven of the sixteen individual checkpoint/K scores matched exactly. Nineteen
individual episode returns and 23 first-arrival steps differed. The largest
individual checkpoint/K score difference was approximately 0.21013 points.
These checks support reproduction of the protocol and close CPU agreement with
the archived outcomes; they do not establish bitwise trajectory identity.

The [check record](../reference_results/maze2d_umaze/release_checks.json)
and [evaluation results](../reference_results/maze2d_umaze/release_validation.json)
record the validation scope. The evaluation record compares a CPU run with the
archived GPU outcomes. Identical model weights and evaluation settings do not
imply bitwise-identical trajectories across numerical backends.

## Checks still outside this snapshot

The install recipe has not been executed from scratch on a newly provisioned
machine or container. Its direct dependency versions are available and the
public D4RL wheel works in the existing environment, but the environment is not
a full historical transitive lockfile.

The full 300-epoch forward-model and 2000-epoch Inverter training campaigns were
not rerun. GPU execution and the paper's inference-time measurements were not
revalidated. The short CPU training runs check that the portable training
pipeline works; they are not new scientific performance results.
