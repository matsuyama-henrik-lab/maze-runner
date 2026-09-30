# Dojo — train a cat to solve mazes 🐱

Teaching prototype for the 専門セミナー. Students train reinforcement-learning
agents (弟子) in the **dojo** (this Python package) and later let them compete in
the **arena** (a Godot game).

A cat walks through a grid maze, finds a key 🗝️, opens a door 🚪, avoids
traps 🔺 and reaches the exit 🏁. The agent only sees a small window around
itself and learns with PPO, written in plain PyTorch so you can read the whole
algorithm.

| Notebook | Colab |
|---|---|
| 01 — mazes, environment, viewer, baselines | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matsuyama-henrik-lab/maze-runner/blob/main/notebooks/01_maze_and_viewer.ipynb) |
| 02 — training with PPO, evaluation, export | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matsuyama-henrik-lab/maze-runner/blob/main/notebooks/02_train.ipynb) |

Everything runs on a normal laptop **CPU**; no GPU needed.

## Quickstart

### Google Colab

Click a badge above. The first code cell installs the package
(`%pip install git+https://github.com/matsuyama-henrik-lab/maze-runner`). Then run
the cells from top to bottom.

### Your own PC (VS Code / Jupyter)

Needs Python 3.10 or newer.

```bash
git clone https://github.com/matsuyama-henrik-lab/maze-runner.git
cd maze-runner
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# PyTorch CPU build (small download), then this package
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

pytest -q                            # everything OK? (< 30 s)
```

Open `notebooks/01_maze_and_viewer.ipynb` in VS Code or Jupyter and choose the
`.venv` Python as kernel.

### Server (JupyterHub / Slurm)

Install once into your own environment (same commands as on your PC), e.g.
into `~/dojo-venv`:

```bash
python -m venv ~/dojo-venv
source ~/dojo-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "git+https://github.com/matsuyama-henrik-lab/maze-runner" ipykernel
python -m ipykernel install --user --name dojo --display-name "dojo"   # for JupyterHub
```

In JupyterHub, choose the kernel "dojo". For longer training runs, use a Slurm
job. Save this as `train.sbatch` and start it with `sbatch train.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=dojo
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=dojo_%j.log

source ~/dojo-venv/bin/activate
python -m dojo.train --steps 1000000 --key-door --out runs/keydoor_$SLURM_JOB_ID
```

Several seeds at once: add `#SBATCH --array=0-4` and use
`--seed $SLURM_ARRAY_TASK_ID --out runs/keydoor_seed$SLURM_ARRAY_TASK_ID`.
A GPU does not help here: the networks are tiny, one CPU core is fastest.

## Commands

The same commands work in a terminal, in a notebook cell (with `!` in front)
and in a Slurm script.

```bash
# Train (about 20 s for 300k steps, 1–2 min for 1M steps on a laptop)
python -m dojo.train --steps 300000 --out runs/demo
python -m dojo.train --steps 1000000 --key-door --out runs/keydoor
python -m dojo.train --steps 1000000 --width 11 --height 11 --loops 0.1 --key-door --traps 3 --out runs/full
python -m dojo.train --steps 1000000 --maze demo --explored --out runs/demo_maze
python -m dojo.train --help            # all options

# Serve an agent to the Godot arena (see godot/README.md)
python -m dojo.bridge --weights runs/keydoor/weights.json --key-door
```

A training run writes plain files into `--out`:

| File | Content |
|---|---|
| `checkpoint.pt` | actor + critic weights and the training config |
| `weights.json` | actor weights + env config, for the arena |
| `curve.csv` | training curve (return, success rate, ... per update) |
| `eval.json` | results on 50 new mazes, next to the wall follower and a random agent |

## What is in the package

| File | Content |
|---|---|
| `dojo/maze.py` | maze generation, tile table, validity check, ASCII/JSON |
| `dojo/env.py` | `MazeEnv` (Gymnasium): actions, observation, rewards (`REWARDS`) |
| `dojo/agents.py` | baselines: random agent, wall follower |
| `dojo/viewer.py` | live viewer, HTML replay, GIF export |
| `dojo/train.py` | our PPO, evaluation, weight export |
| `dojo/bridge.py` | TCP server for the Godot arena |

`MazeEnv` is a standard Gymnasium environment, so other RL libraries work too.
Notebook 02 has an optional section with
[Stable Baselines3](https://stable-baselines3.readthedocs.io/)
(`pip install "dojo[sb3] @ git+https://github.com/matsuyama-henrik-lab/maze-runner"`).

## Development setup

For working on the package itself. Uses [uv](https://docs.astral.sh/uv/) and
Python 3.12 (the same version as Google Colab). The virtual environment lives in
`.venv/` inside the project.

```bash
git clone git@github.com:matsuyama-henrik-lab/maze-runner.git
cd maze-runner

# 1. Virtual environment with Python 3.12 (uv downloads Python if needed)
uv venv --python 3.12 .venv

# 2. PyTorch, CPU-only build (much smaller than the default CUDA build)
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu

# 3. This package (editable) + development tools + Stable Baselines3
uv pip install --python .venv/bin/python -e ".[dev,sb3]"

# 4. Optional: make the venv available as a Jupyter kernel called "dojo (venv)"
.venv/bin/python -m ipykernel install --user --name dojo-venv --display-name "dojo (venv)"

# 5. Check that everything works (< 30 s)
.venv/bin/python -m pytest -q
```

Set your commit email for this repository (GitHub "noreply" address):

```bash
git config user.email "59247064+h-skibbe@users.noreply.github.com"
```

### Notebooks

Notebooks are stored as jupytext percent-format `.py` files, paired with
`.ipynb`. Edit the `.py` file, then update the `.ipynb`:

```bash
.venv/bin/jupytext --sync notebooks/*.py
```

Commit the `.ipynb` files **without outputs** (jupytext writes them that way).
Run a notebook from the command line (the result goes to a separate file):

```bash
.venv/bin/jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=dojo-venv \
    --output /tmp/01_executed.ipynb notebooks/01_maze_and_viewer.ipynb
```
