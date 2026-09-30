# Godot arena — protocol (placeholder)

The arena will be a Godot game that shows the maze and the cat, and later
lets humans play against trained agents. **There is no Godot code yet**;
this file describes how Godot talks to Python.

**Python owns the game rules.** Godot never decides what happens; it sends
messages to `dojo/bridge.py` and draws the state it gets back.

## Start the Python side

```bash
python -m dojo.bridge                                        # wall follower
python -m dojo.bridge --weights runs/demo/weights.json       # a trained agent
python -m dojo.bridge --weights runs/demo/weights.json --key-door --width 11 --height 11
python -m dojo.bridge --maze demo                            # always the demo maze
```

The server listens on `127.0.0.1:11008` (only this computer). It serves one
client at a time; when Godot disconnects, the next one can connect.

## Connection

- TCP, **one JSON message per line** (UTF-8, ending with `\n`).
- Every message from Godot gets exactly one reply.
- Set `TCP_NODELAY` on the Godot side too (`StreamPeerTCP.set_no_delay(true)`),
  otherwise small messages can be delayed by ~40 ms.
- A round trip takes about 1 ms on a laptop.

## Messages

| Godot → Python | Python → Godot |
|---|---|
| `{"type": "reset", "seed": 3}` | `{"type": "reset", "maze": {...}, "state": {...}, "config": {...}, "obs_layout": {...}}` |
| `{"type": "step"}` — the agent chooses | `{"type": "step", "action": 0, "state": {...}, "reward": -0.01, "done": false}` |
| `{"type": "step", "action": 2}` — a human chose | same as above |
| anything invalid | `{"type": "error", "message": "..."}` |

- `seed` is optional. The same seed always gives the same maze
  (`dojo.maze.generate(..., seed=3)`). With `--maze` the maze is fixed.
- After `"done": true`, send a new `reset`.

### Maze

```json
{"width": 11, "height": 9, "rows": ["###########", "<...#.....#", "..."]}
```

Each character is one cell. `x` = column, `y` = row, origin top-left (the same as a
TileMap), so the cell at `(x, y)` is `rows[y][x]`.

| Symbol | Tile | Rule |
|---|---|---|
| `#` | wall | blocks |
| `.` | floor | |
| `<` | entry | blocks (the agent starts on the cell next to it) |
| `>` | exit | episode ends, +1.0 |
| `k` | key | picked up when entered (+0.2) |
| `D` | door | blocks without key; with the key it opens (+0.2) and becomes floor |
| `^` | trap | −0.5 every time it is entered; the episode continues |

The rewards are in `REWARDS` in `dojo/env.py`. Colors and emoji for each tile are in
`TILES` in `dojo/maze.py`.

### State

```json
{"x": 1, "y": 1, "direction": 1, "has_key": false, "door_open": false,
 "visited": [[1, 1], [2, 1]], "steps": 2, "trap_hits": 0}
```

- `direction`: 0 = north (up), 1 = east (right), 2 = south (down), 3 = west (left).
- The key is gone when `has_key` is true; the door is gone when `door_open` is true.

### Actions

`0` = forward, `1` = turn left, `2` = turn right.

## Later: the agent inside Godot

Later the trained agent may run directly in GDScript (no Python needed).
Everything for that is in `weights.json`:

```json
{"config": {"view": 5, "compass": true, "explored": false, "obs_version": 1},
 "obs_layout": {"view": [0, 150], "has_key": [150, 151], "compass": [151, 153]},
 "actions": ["forward", "turn left", "turn right"],
 "activation": "tanh",
 "layers": [{"W": [[...]], "b": [...]}, {"W": ..., "b": ...}, {"W": ..., "b": ...}]}
```

1. **Build the observation** exactly as described in the docstring of `dojo/env.py`
   (view window rotated to the agent's direction, one-hot channels
   `wall, key, door, trap, exit, visited`, then `has_key`, `compass`, and `explored`
   if switched on). `obs_layout` gives the index range of each part.
2. **Forward pass**: `x = W @ x + b` for each layer (`W` has shape `(n_out, n_in)`),
   `tanh` after every layer except the last. The last layer gives 3 logits.
3. **Choose the action by sampling**: `p = softmax(logits)`, then pick an action
   randomly with these probabilities. (Always taking the largest logit can make the
   agent walk in circles.) The reference is `forward()` and `weights_policy()` in
   `dojo/train.py`.
4. **Check the config**: refuse to load weights whose `config` does not match the
   arena's settings (different `view`, `compass`, `explored` or `obs_version`).
