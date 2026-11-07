# Install the Maze2D U-Maze environment

This example reproduces **legacy D4RL `maze2d-umaze-v1`** on Linux x86_64.
Use a separate environment: the modern `mujoco` Python package, Gymnasium,
and Minari do not replace the simulator/wrapper combination used here.

The direct versions below were audited in the working research environment on
2026-11-07. They are a reproducible starting recipe, **not a complete historical
lockfile** of every transitive dependency used in the original evaluation.

| Component | Version |
| --- | --- |
| Python | 3.9.25 |
| Gym | 0.23.1 |
| D4RL | 1.1, exact PyPI wheel pinned by SHA-256 |
| `mujoco-py` | 2.1.2.14 |
| Native MuJoCo | 2.1.0 (`mujoco210`) |
| PyTorch | 2.8.0 |
| NumPy / SciPy | 1.24.4 / 1.11.4 |
| h5py / Cython | 3.14.0 / 0.29.37 |

Evaluation works on CPU. The archived timing measurements used an NVIDIA A40
with 48 GB, and full training is intended for a CUDA GPU. The 48 GB figure is
the paper's hardware specification, not a measured minimum memory requirement.
The bundled checkpoints occupy approximately 37 MiB; training data and Python
packages require additional disk space.

## 1. System dependencies

On Ubuntu 24.04, install the compiler and OpenGL/OSMesa development libraries:

```bash
sudo apt-get update
sudo apt-get install -y build-essential curl patchelf \
  libgl1 libgl1-mesa-dev libosmesa6-dev libglew-dev libglfw3 libglfw3-dev
```

On a managed server, use the corresponding preinstalled packages or ask its
administrator. No desktop session or display is needed for the numerical
U-Maze evaluation. `mujoco-py` builds a native extension on first import, so
compiler and GL libraries are still required. See the upstream
[`mujoco-py` installation notes](https://github.com/openai/mujoco-py#install-mujoco).

## 2. Create the Python environment

Run these commands from a new checkout:

```bash
git clone https://github.com/neuromentum/inverter.git
cd inverter
conda create -n inverter-maze2d python=3.9.25 pip -y
conda activate inverter-maze2d

python -m pip install setuptools==80.9.0 wheel==0.45.1 \
  Cython==0.29.37 numpy==1.24.4
```

Choose **one** PyTorch build before installing the remaining requirements.
For the initial CPU tutorial:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

For a CUDA machine with a compatible NVIDIA driver, the audited research
installation uses the CUDA 12.8 build:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

The [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/#v280)
list the available builds. Check `nvidia-smi` and driver compatibility before
choosing a CUDA wheel. A system-wide CUDA toolkit is not needed for the CPU path.

```bash
python -m pip install --no-build-isolation -r requirements/maze2d.txt
python -m pip install --no-deps -r requirements/d4rl.txt
python -m pip install --no-deps --no-build-isolation -e .
```

D4RL's dependency list includes other benchmark suites. The `--no-deps` install
keeps this tutorial focused on PointMaze; its required packages are installed
explicitly. Our adapter imports `d4rl.pointmaze` explicitly, so a missing optional
suite does not prevent U-Maze registration. No edits to installed D4RL files are
required.

## 3. Install native MuJoCo and configure the loader

The native simulator is separate from the `mujoco-py` wheel. The setup helper
downloads the official [MuJoCo 2.1.0 release](https://github.com/google-deepmind/mujoco/releases/tag/2.1.0),
checks its archive hash, and installs it only when the destination is absent:

```bash
bash scripts/setup_maze2d.sh --download-mujoco
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset MAZE2D_BODY_POS MAZE2D_CENTERED
```

Run the exports in each new shell, or put them in this conda environment's
activation script. If MuJoCo 2.1.0 is already installed elsewhere, set
`MUJOCO_PY_MUJOCO_PATH` before invoking the helper. It will not overwrite an
existing installation.

For a headless CPU-rendering build, including on a machine without an NVIDIA
driver:

```bash
export MUJOCO_PY_FORCE_CPU=1
```

This selects `mujoco-py`'s OSMesa extension; it does not disable CUDA computation
in PyTorch. Alternatively, leave it unset to use `mujoco-py`'s NVIDIA extension.
If that extension requests a driver-library directory, add the exact directory
printed by the error to `LD_LIBRARY_PATH` (the research host uses
`/usr/lib/nvidia`). No rendering is invoked by this tutorial's evaluator.

## 4. Check the task before evaluating a model

```bash
python scripts/check_maze2d.py --strict-versions
```

The check verifies native MuJoCo 2.1.0, maze geometry, fixed goal `(1, 1)`,
300-step horizon, simulator timestep, seed-777 reset state, and a simulator
transition. It prints a JSON report. `--strict-versions` additionally requires
the audited direct package versions and Python 3.9. The check does not download
training data or run an Inverter.

Deprecation notices from Gym and `mujoco-py` are expected for this deliberately
preserved benchmark stack. A task mismatch or failed numerical check should be
resolved before comparing scores. See the [complete evaluation protocol](maze2d-protocol.md).

## 5. Training data

The evaluator needs no offline dataset. For training, run:

```bash
python examples/maze2d_umaze/prepare_data.py
python scripts/check_maze2d.py --dataset
```

The adapter uses the original U-Maze dataset over HTTPS. D4RL caches it at
`~/.d4rl/datasets/maze2d-umaze-sparse-v1.hdf5`; set `D4RL_DATASET_DIR` before
starting Python to use another cache directory. The preparation script writes
the five extracted arrays to `data/maze2d_v1/umaze/` and checks their hashes.

- [Original HDF5 download](https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5)
- File size: 34,720,209 bytes; 1,000,000 transitions; 4 state and 2 action dimensions.
- SHA-256: `f2d327d95067d53e274ac2fe876b740ed7883d0a9738bd5b447635cdb786bfb7`.
- The saved timeout/terminal segmentation yields 12,460 segments.

The optional `--dataset` check validates an already cached file and never
downloads it. It also accepts an explicit HDF5 path.

D4RL publishes its code under Apache-2.0 and, unless otherwise noted, its datasets
under CC BY 4.0. Cite Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and
Sergey Levine, *D4RL: Datasets for Deep Data-Driven Reinforcement Learning* (2020).
See [D4RL's citation and license instructions](https://github.com/Farama-Foundation/D4RL#licenses).

## Common setup problems

| Symptom | Check |
| --- | --- |
| `No module named inverter` | Activate the intended environment and run the editable install from its repository root. |
| `maze2d-umaze-v1` is not registered | Use this example's adapter and the pinned D4RL wheel; do not substitute a newer benchmark package. |
| MuJoCo library / GLFW / `-lGL` error | Check native MuJoCo's path, loader exports, compiler, and system GL development packages. |
| Cython callback compilation error | Confirm `Cython==0.29.37`; avoid Cython 3 for this legacy wrapper. |
| Missing NVIDIA libraries | Use the OSMesa CPU-rendering option or a working driver-library path. |
| CUDA initialization failure | Start with `--device cpu`; check the NVIDIA driver before using a CUDA PyTorch build. |
| Reset-state or body-origin mismatch | Unset geometry overrides and use stock D4RL 1.1 with MuJoCo 2.1.0. |

The initial repository validation uses the existing research environment on
CPU. The environment check also passes with the unmodified, checksum-verified
D4RL 1.1 wheel loaded in that environment. The recipe above has not yet been
tested end to end in a freshly provisioned machine or container; it should not
be treated as a fully locked container image.
