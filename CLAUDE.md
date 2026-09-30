# Dojo — RL maze prototype for the 専門セミナー

Teaching prototype for a 3rd-year undergraduate seminar (starting April 2027).
Students train reinforcement-learning agents ("弟子") in the **dojo** (Python) and
later let them compete in the **arena** (a Godot game). A cat walks through a grid
maze, finds a key, opens a door, avoids traps and reaches the exit.

## Guiding principles (most important section)

- **Not over-engineered.** Few files, no class hierarchies, no plugin systems, no
  config frameworks (argparse is enough). Prefer a plain function over a class.
- **Readable by 3rd-year students.** Short functions, clear names, comments that
  explain *why*. Students will read and modify this code. Avoid clever tricks.
- **Easy to maintain by one person.** Few, pinned dependencies. No code
  generation, no metaprogramming.
- **Runs identically** in Google Colab, a local VS Code/Jupyter setup, and a
  JupyterHub/Slurm GPU server — same install, same commands. **CPU by default**;
  never assume CUDA.
- **No display windows.** No pygame/tkinter. Rendering = NumPy/PIL images or
  HTML/JS in the notebook.
- Build in small steps. After each step: run the tests, show a short demo, stop
  and let me review before continuing.

## File layout (keep it this small)

```
dojo/
  CLAUDE.md
  README.md            # install + quickstart (Colab / local / server)
  pyproject.toml       # installable: pip install git+https://.../dojo
  dojo/
    __init__.py
    maze.py            # maze generation, placing key/door/trap, checks, ASCII/JSON I/O
    env.py             # Gymnasium environment (MazeEnv)
    agents.py          # baselines: random agent, wall follower
    viewer.py          # rendering, live notebook viewer, HTML replay player, GIF export
    train.py           # our own PPO (plain PyTorch), evaluation, weight export
    bridge.py          # placeholder: localhost socket server for the Godot arena
  notebooks/
    01_maze_and_viewer.py   # jupytext (percent format), paired with .ipynb
    02_train.py
  tests/
    test_dojo.py
  godot/
    README.md          # placeholder: protocol summary, arena to be built later
```

Ask before adding files beyond this list.

## Dependencies

numpy, gymnasium, torch, ipywidgets, pillow.
Optional extra `sb3`: stable-baselines3 (`pip install dojo[sb3]`) — the core
package must work and all core tests must pass without it.
Dev: pytest, jupytext. Python >= 3.10. Nothing else without asking.

## Maze representation

- A maze is a list of strings (rows), roguelike ASCII symbols:
  `#` wall, `.` floor, `<` entry, `>` exit, `k` key, `D` door, `^` trap.
  The agent start is the floor cell next to the entry; the agent itself is not
  stored in the maze (it is part of the env state). `@` is only used when
  printing a maze with the agent.
- Coordinates: `x` = column, `y` = row, origin top-left (same as a Godot TileMap).
- Directions: 0=N, 1=E, 2=S, 3=W.
- JSON format (for Godot and saved test mazes): `{"width": W, "height": H, "rows": [...]}`
  — simply the ASCII rows, so it stays human-readable.
- Include this **demo maze** as a constant (it is the one on the seminar web page):

```
###########
<...#.....#
#.#.#.#.#.#
#.#...#.#.#
#.#####.#.#
#...^...#.#
###.#.#####
#k..#....D>
###########
```

## maze.py

- **One tile table** at the top of the file (symbol, name, color, emoji). All other
  code (observation channels, rendering, viewer legend) is derived from this table,
  so adding a tile type later = one table row + its rule. See "Future extensions".
- `generate(width, height, seed, loops=0.0, key_door=False, traps=0)`:
  depth-first-search "perfect maze" on odd dimensions; `loops` = fraction of
  extra walls removed to create cycles.
- Key/door placement: door on a cell that every path from start to exit must
  pass (a bottleneck); key reachable from the start **without** passing the door.
  Traps on floor cells, never on the entry/exit/key/door, never blocking all paths.
- `check(maze)`: BFS-based validity check (exit reachable, key before door, ...).
  Every generated maze must pass it.
- `to_ascii`, `from_ascii`, `to_json`, `from_json`.
- Seeded `numpy.random.Generator` only — no global random state.

## env.py — MazeEnv (Gymnasium)

- Must be a **standard Gymnasium env** and pass
  `gymnasium.utils.env_checker.check_env` (add a test). Then any RL library
  (e.g. Stable Baselines3) can be used on it later without changing env.py.
- Actions (discrete, egocentric): 0 = forward, 1 = turn left, 2 = turn right.
- Observation (flat float32 vector, documented in the docstring with its exact
  layout, because Godot must reproduce it later):
  - local `view × view` window (default 5), **rotated to the agent's facing
    direction**, one-hot channels: wall, key, door, trap, exit, visited;
    cells outside the maze count as wall
  - `has_key` flag
  - optional compass to the exit (2 values, egocentric), `compass=True` by default
- Rewards: all values in **one dict at the top of the file** so students can edit
  them — e.g. step −0.01, key +0.2, door opened +0.2, trap −0.5 (episode
  continues, so a shortcut over a trap can be worth it), exit +1.0.
- Episode ends at the exit; truncated after `max_steps` (default 4 × number of cells).
- Walking into a closed door without the key = blocked, like a wall. With the key,
  the door opens and becomes floor.
- `reset(seed)` generates a new maze from the generator settings, or uses a fixed
  maze if one was given (`MazeEnv(maze=DEMO_MAZE)`).
- `render()` supports `"rgb_array"` (one colored block per cell) and `"ansi"`.
- `state()` returns a small JSON-able dict (position, direction, has_key,
  door_open, visited, steps) — used by the replay viewer and the Godot bridge.

## agents.py

- `random_agent(obs)` and a **wall follower** (right-hand rule) implemented from
  the env state. The wall follower is the baseline every learned agent is
  compared against.

## viewer.py

- `LiveViewer`: ipywidgets `Image` updated in a loop (PIL, nearest-neighbour
  upscale). Works in Jupyter, VS Code notebooks and Colab.
- `record_episode(env, policy, seed) -> dict`: maze + list of states + rewards.
- `replay_html(episode)`: self-contained HTML/JS canvas player shown with
  `IPython.display.HTML`; emoji symbols (🐱 🗝️ 🚪 🔺 🏁), smooth movement between
  cells, play/pause/speed buttons. No external JS libraries.
- `save_gif(episode, path)`: animated GIF via Pillow (for the seminar web page).
  Optional sprite PNGs from an `assets/` folder may be added later; start with
  simple colored tiles.

## train.py

- `python -m dojo.train --steps 300000 --n-envs 16 --out runs/demo [--key-door] [--loops 0.1]`
  — same command in a terminal, a notebook cell (`!python -m ...`) or an sbatch script.
- **Main training code: our own PPO implementation in plain PyTorch.** Students must be
  able to read the whole learning algorithm top to bottom (CleanRL single-file
  style): rollout collection, GAE advantages, clipped policy loss, value loss,
  entropy bonus, minibatch updates. Every step commented, hyperparameters as a
  small dataclass or argparse defaults at the top.
- Network: separate actor and critic, each a plain `nn.Sequential` MLP
  (64, 64, tanh). Discrete actions via logits + `torch.distributions.Categorical`.
- Parallel environments: a simple list of `MazeEnv` instances stepped in a loop
  (or `gymnasium.vector.SyncVectorEnv`) — no subprocesses.
- `device="cpu"` unless `--device` is given.
- Correctness checks: our PPO must clearly beat the random agent on small
  perfect mazes within a few minutes on a laptop, and reach roughly the same
  success rate as SB3 PPO on the same mazes and step budget; report both in step 4.
- **SB3 as an optional alternative** (for students who prefer a well-documented
  library, and as a reference to validate our PPO):
  - Only imported inside the SB3-specific functions, so the core works without it.
  - `export_weights_sb3(model, path)`: exports an SB3 `MlpPolicy` actor to the
    **same** weights JSON format, so SB3-trained agents play in the same arena.
  - Notebook 02 gets one short, clearly marked optional section "Training with
    Stable Baselines3" (a few lines: `PPO("MlpPolicy", env).learn(...)`, then
    evaluate + export).
  - No separate SB3 training script; no `--algo` switch in `train.py`.
- `evaluate(policy, n_mazes=50, seed=...)` on **held-out seeds** (never seen in
  training): success rate, mean steps, trap hits — reported next to the wall
  follower.
- `export_weights(model, path)`: actor MLP weights to JSON
  (`[{"W": ..., "b": ...}, ...]`) for the GDScript forward pass later.
  Include a tiny NumPy `forward(weights, obs)` that reproduces the PyTorch
  actor's greedy action; a test checks that both agree.
- Outputs are plain files in `--out`: checkpoint (`torch.save` of the state
  dicts + config), weights JSON, a CSV of the
  training curve, evaluation summary.

## bridge.py — placeholder for the Godot arena

Python owns the game rules; Godot only displays and (later) lets humans play.
Keep this minimal and well commented:

- TCP server on `127.0.0.1:11008`, **`TCP_NODELAY` on**, one JSON message per line.
- Messages (Godot → Python / Python → Godot):
  - `{"type": "reset", "seed": 3}` → `{"maze": <maze JSON>, "state": <state>}`
  - `{"type": "step"}` → policy chooses: `{"action": a, "state": ..., "reward": r, "done": b}`
  - `{"type": "step", "action": a}` → a human chose the action (same reply)
- `python -m dojo.bridge --weights runs/demo/weights.json` serves a trained agent;
  without weights it uses the wall follower.
- A tiny Python test client in the tests stands in for Godot.
- Document the protocol in `godot/README.md`. No Godot code yet.

## Tests (pytest, fast: whole suite < 30 s on a laptop)

- Every generated maze passes `check()` (many seeds, with/without loops, key/door, traps).
- Wall follower solves perfect mazes without key/door.
- Observation shape/layout; rotation is correct (a wall ahead appears in the same
  observation slot regardless of the facing direction).
- Door blocks without key, opens with key.
- Exported NumPy forward pass == PyTorch actor greedy action.
- PPO smoke test: a short training run on a tiny fixed maze improves the mean return.
- SB3 export gives the same greedy action as the SB3 model (skipped via
  `pytest.importorskip` if SB3 is not installed).
- Bridge round trip with the test client.

## Build order (stop for review after each step)

1. `maze.py` + tests; print a few generated mazes and the demo maze.
2. `env.py` + `agents.py` + tests; random agent vs wall follower on 50 mazes.
3. `viewer.py` + notebook 01: live viewer, HTML replay, GIF of the wall follower
   on the demo maze (**the GIF is needed for the web page by Oct 15**).
4. `train.py` + notebook 02: PPO on small perfect mazes → then loops → then
   key/door. Report training time and success rates. Then the optional SB3
   section + export, and compare our PPO against SB3 on the same mazes.
5. `bridge.py` + `godot/README.md`.
6. README with install/quickstart for Colab, local and server.

## Future extensions — do NOT implement yet, but keep the design open for them

Planned as higher difficulty levels ("belts") or student group projects:

- **Jump**: extra action; moves two cells over a trap or a new "gap" tile (gaps
  cannot be walked on). Small extra cost so the agent does not jump everywhere.
- **Several keys with matching colored doors**: `has_key` becomes an inventory
  vector; the validity check becomes a BFS over (position, inventory), and keys
  may lie behind other doors.
- **Speed boost item**: for the next N steps, "forward" moves two cells; the
  remaining boost counter is part of the observation.
- **Far sight item**: after pickup, a ray along the facing direction reports the
  distance to the first non-floor cell and its type (one-hot).

What this means for the prototype design **now**:

- Tile types only in the tile table (see maze.py); rules for each tile in one
  clearly separated place in env.py.
- Features are switched on by env arguments, e.g.
  `MazeEnv(jump=False, key_colors=1, boosts=0, far_sight=False)`. With all
  features off, the env must behave exactly like the basic version. For the
  prototype, only add the arguments that are actually implemented.
- **Fixed observation size per configuration**: collectable upgrades have their
  observation slots always present, zero-filled plus an "active" flag until picked
  up. The network input size never changes during an episode.
- Build the observation from small named parts (view window, inventory, compass,
  ...) and provide `obs_layout(config)` that returns names and index ranges — used
  in docs, tests and later by Godot.
- **The configuration travels with the agent**: the exported weights JSON and the
  bridge `reset` reply contain the env config (feature flags, view size, obs
  layout version). Loading an agent with a mismatching config raises a clear error.

## Conventions

- Comments and docstrings in English, simple wording (students are not native
  speakers). Japanese is fine in user-facing notebook markdown.
- Type hints on public functions; no heavy typing gymnastics.
- Notebooks are jupytext percent-format `.py` files paired with `.ipynb`; edit the
  `.py` side.
