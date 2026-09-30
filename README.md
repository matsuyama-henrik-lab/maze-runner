# Dojo

Train reinforcement-learning agents (弟子) to solve grid mazes.
Teaching prototype for the 専門セミナー.

Install and quickstart instructions follow (Colab / local / server).

## Development setup (local PC)

Uses [uv](https://docs.astral.sh/uv/) and Python 3.12 (the same version as
Google Colab). The virtual environment lives in `.venv/` inside the project.

```bash
git clone <repository-url> maze-runner
cd maze-runner

# 1. Virtual environment with Python 3.12 (uv downloads Python if needed)
uv venv --python 3.12 .venv

# 2. PyTorch, CPU-only build (much smaller than the default CUDA build)
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu

# 3. This package (editable) + development tools (pytest, jupytext, ipykernel, nbconvert)
uv pip install --python .venv/bin/python -e ".[dev]"

# 4. Optional: make the venv available as a Jupyter kernel called "dojo (venv)"
.venv/bin/python -m ipykernel install --user --name dojo-venv --display-name "dojo (venv)"

# 5. Check that everything works (< 30 s)
.venv/bin/python -m pytest -q
```

Optional, for comparing against Stable Baselines3: `uv pip install --python .venv/bin/python -e ".[dev,sb3]"`.

Without uv: `python3.12 -m venv .venv`, then the same commands with
`.venv/bin/pip install ...` instead of `uv pip install --python .venv/bin/python ...`.

### Notebooks

Notebooks are stored as jupytext percent-format `.py` files, paired with
`.ipynb`. Edit the `.py` file, then update the `.ipynb`:

```bash
.venv/bin/jupytext --sync notebooks/*.py
```

Run a notebook from the command line (the result goes to a separate file):

```bash
.venv/bin/jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=dojo-venv \
    --output /tmp/01_executed.ipynb notebooks/01_maze_and_viewer.ipynb
```
