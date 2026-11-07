# Tutorial 1: turn a goal into actions in U-Maze

A point mass must reach a known target in a U-shaped corridor. Its state is
`[x, y, vx, vy]`, and each action has two components. This example introduces
inverse learning with a learned forward model and a goal-conditioned Inverter.

## 1. Set up and check D4RL

Follow the [environment guide](../../docs/maze2d-environment.md), then work from
the repository root with the `inverter-maze2d` environment active. The legacy
benchmark setup is part of the experiment: newer simulator wrappers can change
reset states, dynamics, or scoring.

```bash
conda activate inverter-maze2d
python scripts/check_maze2d.py
```

## 2. Try a saved Inverter

Start with a few CPU episodes. This checks the complete controller/simulator
path without training or downloading the offline dataset:

```bash
python examples/maze2d_umaze/evaluate.py \
  --model default --replan-every 64 --episodes 3 \
  --eval-seed 777 --device cpu --output outputs/umaze-smoke
```

A full paper comparison evaluates four archived models and four execution
lengths on 100 episodes each:

```bash
python examples/maze2d_umaze/evaluate.py \
  --model all --replan-every 16 32 64 128 --episodes 100 \
  --eval-seed 777 --device cuda:0 --output outputs/umaze-paper
```

The `--output` argument names a directory. Each condition writes its own
`results.json`, and `summary.json` collects the scores.

Use `--device cpu` if no CUDA GPU is available. Runtime measurements will depend
on the machine. See the [protocol and expected scores](../../docs/maze2d-protocol.md)
for exactly what this evaluation measures.

The archived checkpoints are in `checkpoints/`. Their original configurations,
checkpoint identities, and hashes are in `configs/`. The forward models are
provided for training and inspection; evaluation uses only the Inverter.

## 3. Prepare the offline training data

Training uses the original `maze2d-umaze-v1` D4RL data set, containing one million
transitions. Evaluation alone does not require it. This step downloads the
original data if it is not already cached:

```bash
python examples/maze2d_umaze/prepare_data.py
```

The data cache is separate from the Git repository. The prepared arrays are
written to `data/maze2d_v1/umaze/` by default. The environment guide records the
original download and checksum.

## 4. Learn what actions do

Train a forward model to predict a 16-step state sequence from the initial state
and 16 actions. This is supervised prediction using the offline data:

```bash
python examples/maze2d_umaze/train_forward.py \
  --maze umaze --gpu 0 --seed 42 \
  --epochs 300 --batch_size 512 --lr 3e-4 \
  --chunk_length 16 --d_model 128 --n_heads 4 --n_layers 4 \
  --ff_mult 4 --dropout 0.1 --s0_noise 0.01 \
  --n_train 500000 --n_val 50000 \
  --name umaze_forward --out_dir outputs/maze2d_umaze/forward
```

The state-noise setting helps the model handle small state deviations. Inspect
the validation prediction error before using a new forward model for inverse
learning: an Inverter can exploit errors in an inaccurate model.

## 5. Learn to choose actions

Freeze the forward model. Train an Inverter that receives the current state
and target state and emits 128 actions. Gradients pass through eight successive
16-step forward-model chunks to improve the predicted outcome:

```bash
python examples/maze2d_umaze/train_inverse.py \
  --maze umaze --gpu 0 --seed 123 \
  --epochs 2000 --batch_size 128 --lr 3e-4 \
  --segment_length 16 --d_model 192 --n_heads 6 --n_layers 4 --d_ff 576 \
  --n_train 100000 --n_val 10000 \
  --lambda_terminal 0 --lambda_dense 5 --lambda_boundary 5 \
  --boundary_mode binary --no_episode_aware \
  --fm_path outputs/maze2d_umaze/forward/best_model.pt \
  --name umaze_inverter --out_dir outputs/maze2d_umaze/inverter
```

The objective rewards predicted progress toward the goal and penalizes leaving
the region supported by offline data. It does not require optimal-action labels.
The saved checkpoint is selected by validation loss. The inherited training
monitor also runs random-goal diagnostic episodes; these diagnostics use a
different controller protocol and are not the fixed-goal paper evaluation in
steps 2 and 6.

These are the full archived training budgets, not quick smoke runs. The paper
reports approximately seven hours for a U-Maze forward-model/Inverter training
pair on one 48 GB A40; time and memory usage depend on the implementation and
machine. Four checkpoint families require four paired training runs.

The archived sampling and seed choices have important qualifications; read
[the training details](../../docs/maze2d-protocol.md#training-details-that-matter)
before expecting exact agreement from retraining.

## 6. Evaluate your own Inverter

```bash
python examples/maze2d_umaze/evaluate.py \
  --checkpoint outputs/maze2d_umaze/inverter/best_model.pt \
  --replan-every 64 --episodes 100 --eval-seed 777 \
  --device cuda:0 --output outputs/umaze-trained
```

Separate the 128-action prediction horizon from the number K of actions you
execute. Reducing K adds feedback and more network calls; increasing K spends
longer executing each prediction. Comparing K=16,32,64,128 is a useful first
experiment with this trade-off.

## What to change next

Try a different execution length, inspect forward-model errors away from the
offline data, or change one training-loss weight while keeping the benchmark
protocol fixed. Keep your experimental outputs separate from the archived
reference results so you can compare behavior clearly.
