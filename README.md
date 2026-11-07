# Inverter

**Learn to turn goals into action sequences.**

Inverter is a toolbox and tutorial collection for **inverse learning**: train a
model to predict what actions will do, then train an Inverter to choose actions
through that model. At deployment, the Inverter turns an observed state and a
goal into an action sequence in one forward pass. Executing a short part of that
sequence and planning again lets the controller respond to new observations.

The repository accompanies *Neuro-Inspired Inverse Learning for Planning and
Control* by Maryna Kapitonova and Tonio Ball
([paper](https://arxiv.org/abs/2605.24152)). Its aim is to make the approach useful
beyond the paper: understandable examples, reusable model components, and
explicit training and evaluation protocols.

## Start with Maze2D U-Maze

Our first tutorial controls a point mass in a U-shaped maze. It introduces the
complete workflow with a four-dimensional observed state, two action channels,
and a known target:

1. **Learn a forward model** from an offline data set of states and actions.
2. **Train an Inverter** through the frozen forward model to reach a target.
3. **Act and replan:** emit 128 actions, execute a chosen number, observe the
   resulting state, and repeat.
4. **Evaluate** using the paper's D4RL task, reset seeds, and scoring protocol.

No routing planner or learned forward model is needed during this example's
deployment. The supplied Inverter checkpoints let you start with evaluation
before downloading training data or training your own models.

- [Install and check the benchmark environment](docs/maze2d-environment.md)
- [Follow the U-Maze tutorial](examples/maze2d_umaze/README.md)
- [Understand the evaluation protocol and expected scores](docs/maze2d-protocol.md)

## What is in this first version?

| Component | Purpose |
| --- | --- |
| `src/inverter/models/` | Forward models, Inverter networks, and differentiable training losses |
| `src/inverter/envs/` | The benchmark environment adapter |
| `examples/maze2d_umaze/` | Data preparation, training, evaluation, and the archived checkpoints |
| `requirements/` | The environment recipe for the first tutorial |
| `reference_results/maze2d_umaze/` | Frozen evaluation cases, per-model scores, and protocol |
| `scripts/check_maze2d.py` | Checks for simulator, task, and optional training-data compatibility |
| `docs/` | Setup guidance and reproduction details |

This is an initial release. Maze2D U-Maze is the first worked example; the other
paper experiments are not yet packaged here. Planned additions include
hierarchical routing, AntMaze, the Strategy Inverter, and quantum gate synthesis.

## Original submission archive

The unchanged [original NeurIPS submission code bundle](archive/neurips-2026-submission/README.md)
is also available, including code for Maze2D, AntMaze, AntMan, and quantum gate
synthesis. It preserves the historical submission; the tutorials above provide
the maintained setup and evaluation workflow.

## Reproduction and experimentation

The U-Maze example targets **`maze2d-umaze-v1` in legacy D4RL**, using Gym,
`mujoco-py`, and MuJoCo 2.1.0. The environment guide gives explicit versions and
checks. Substituting Gymnasium or a newer maze benchmark changes the environment
being evaluated.

There are two useful starting points:

- **Evaluate the archived models:** use the supplied checkpoints and the exact
  evaluation protocol. Compare normalized scores against the frozen reference.
- **Train a new model:** follow the archived training recipe, inspect the learned
  behavior, and vary the model or objective. Retraining does not promise identical
  weights; see the documented sampling and random-seed details.

The public evaluator reports fresh runtime measurements. The paper's archived
GPU timing values describe its A40 measurements and timing boundaries; they are
not expected to match every machine.

## Validation of this snapshot

The supplied checkpoint and training-data hashes are verified. The environment
checks pass with the public D4RL wheel, and both training stages pass short CPU
runs. A full 1,600-episode CPU evaluation reproduces the four-model mean scores
within 0.019 normalized-score points of the archived GPU results. Small
trajectory and return differences remain across numerical backends. See the
[validation record](docs/validation.md) for the executed checks and their limits.

## The broader toolbox

The same idea can be used wherever a differentiable model connects controllable
inputs to desired outcomes. The goal is to make forward models, Inverters, task
objectives, and execution policies easy to combine. Hierarchical examples will
show how an upper Inverter can choose goals or operating settings for lower
controllers. Each example will state what it observes, what it predicts, what it
optimizes, and what runs at deployment.

## Citation

```bibtex
@article{kapitonova2026inverter,
  title   = {Neuro-Inspired Inverse Learning for Planning and Control},
  author  = {Kapitonova, Maryna and Ball, Tonio},
  journal = {arXiv preprint arXiv:2605.24152},
  year    = {2026}
}
```

## Third-party software and data

The U-Maze example uses D4RL, Gym, MuJoCo, and PyTorch. Training data are downloaded
from their original source, rather than bundled. D4RL's code and datasets have
separate license terms; see the environment guide and the upstream projects.
