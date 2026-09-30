"""Watching agents: live notebook viewer, HTML replay player and GIF export.

    viewer = LiveViewer();  viewer.play(env, policy)     # live, in a notebook
    episode = record_episode(env, policy, seed=3)        # record once ...
    replay_html(episode)                                  # ... replay in the notebook
    save_gif(episode, "cat.gif")                          # ... or save as a GIF

No display windows: everything is a NumPy/PIL image or HTML/JS.
"""

import html
import io
import json
import time

import numpy as np
from PIL import Image

from dojo import maze as mz
from dojo.env import MazeEnv, draw


def grid_at(maze: list[str], state: dict) -> list[list[str]]:
    """The maze as it looks in `state`: the key is gone once the agent has
    it, the door is gone once it is open."""
    grid = [list(row) for row in maze]
    for row in grid:
        for x, tile in enumerate(row):
            if (tile == mz.KEY and state["has_key"]) or (tile == mz.DOOR and state["door_open"]):
                row[x] = mz.FLOOR
    return grid


def frame(maze: list[str], state: dict, cell_size: int = 12, scale: int = 3) -> Image.Image:
    """One picture of the maze and the agent, as a PIL image.
    Drawn small and then enlarged with nearest-neighbour, so it stays sharp."""
    small = Image.fromarray(draw(grid_at(maze, state), state, cell_size=cell_size))
    return small.resize((small.width * scale, small.height * scale), Image.NEAREST)


# ---------------------------------------------------------------------------
# Live viewer (ipywidgets)
# ---------------------------------------------------------------------------

class LiveViewer:
    """Shows the environment in a notebook and updates the picture in place.

        viewer = LiveViewer()
        viewer.play(env, policy, seed=0)

    Works in Jupyter, VS Code notebooks and Colab.
    """

    def __init__(self, cell_size: int = 12, scale: int = 3):
        import ipywidgets  # only needed in notebooks
        from IPython.display import display

        self.cell_size, self.scale = cell_size, scale
        self.image = ipywidgets.Image(format="png")
        self.label = ipywidgets.Label()
        display(ipywidgets.VBox([self.image, self.label]))

    def update(self, env: MazeEnv, text: str = "") -> None:
        """Draw the current state of `env`."""
        picture = frame(env.maze, env.state(), self.cell_size, self.scale)
        buffer = io.BytesIO()
        picture.save(buffer, format="png")
        self.image.value = buffer.getvalue()
        self.label.value = text

    def play(self, env: MazeEnv, policy, seed: int | None = None, fps: float = 8) -> dict:
        """Run one episode and show every step. Returns a small summary."""
        obs, info = env.reset(seed=seed)
        total, done = 0.0, False
        self.update(env, "step 0")
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total += reward
            done = terminated or truncated
            self.update(env, f"step {env.steps}   return {total:+.2f}"
                             f"   {'🗝️' if env.has_key else ''}"
                             f"   {'solved!' if terminated else ''}")
            time.sleep(1 / fps)
        return {"solved": terminated, "steps": env.steps, "return": total,
                "trap_hits": env.trap_hits}


# ---------------------------------------------------------------------------
# Recording and replay
# ---------------------------------------------------------------------------

def record_episode(env: MazeEnv, policy, seed: int | None = None) -> dict:
    """Run one episode and record it: the maze, every state and every reward.

    Returns a JSON-able dict:
        {"maze": [...], "config": {...}, "states": [s0, s1, ...],
         "actions": [...], "rewards": [...], "solved": bool}
    There is one state more than actions/rewards (the start state).
    """
    obs, info = env.reset(seed=seed)
    episode = {"maze": list(env.maze), "config": dict(env.config),
               "states": [env.state()], "actions": [], "rewards": [], "solved": False}
    done = False
    while not done:
        action = int(policy(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        episode["states"].append(env.state())
        episode["actions"].append(action)
        episode["rewards"].append(float(reward))
        done = terminated or truncated
    episode["solved"] = bool(terminated)
    return episode


def replay_page(episode: dict, cell_size: int = 36) -> str:
    """A complete, self-contained HTML page that replays the episode
    (canvas + JavaScript, no external libraries). Can be saved as .html."""
    tiles = {t["symbol"]: {"name": t["name"], "color": "rgb(%d,%d,%d)" % t["color"],
                           "emoji": t["emoji"]} for t in mz.TILES}
    data = {"maze": episode["maze"], "states": episode["states"],
            "rewards": episode["rewards"], "tiles": tiles, "agent": mz.AGENT["emoji"],
            "cell": cell_size, "directions": mz.DIRECTIONS}
    return _PLAYER_TEMPLATE.replace("__DATA__", json.dumps(data))


def replay_html(episode: dict, cell_size: int = 36):
    """Show the replay player in a notebook.

    The page is put into an <iframe>, so several players in one notebook
    do not disturb each other.
    """
    from IPython.display import HTML  # only needed in notebooks

    width = len(episode["maze"][0]) * cell_size + 40
    height = len(episode["maze"]) * cell_size + 130
    page = html.escape(replay_page(episode, cell_size), quote=True)
    return HTML(f'<div><iframe srcdoc="{page}" width="{width}" height="{height}" '
                f'style="border: none;"></iframe></div>')


# ---------------------------------------------------------------------------
# GIF export
# ---------------------------------------------------------------------------

def save_gif(episode: dict, path: str, cell_size: int = 12, scale: int = 3,
             fps: float = 6, end_pause: float = 1.5) -> None:
    """Save the episode as an animated GIF (e.g. for a web page).
    The last picture stays a little longer (`end_pause` seconds)."""
    frames = [frame(episode["maze"], state, cell_size, scale) for state in episode["states"]]
    durations = [int(1000 / fps)] * len(frames)
    durations[-1] = int(1000 * end_pause)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)


# The HTML/JS replay player. __DATA__ is replaced by the episode as JSON.
_PLAYER_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body { font-family: sans-serif; margin: 8px; background: #fff; color: #222; }
  button, select { font-size: 15px; margin-right: 4px; }
  #info { font-family: monospace; margin: 6px 0; }
  #legend { font-size: 13px; color: #555; }
  #legend span { margin-right: 12px; white-space: nowrap; }
  .swatch { display: inline-block; width: 11px; height: 11px; vertical-align: middle; }
</style>
</head>
<body>
<canvas id="maze"></canvas>
<div>
  <button id="play">▶</button>
  <button id="restart">⏮</button>
  <select id="speed">
    <option value="0.5">0.5x</option><option value="1" selected>1x</option>
    <option value="2">2x</option><option value="4">4x</option>
  </select>
  <input id="slider" type="range" min="0" value="0" step="0.01" style="width: 45%;">
</div>
<div id="info"></div>
<div id="legend"></div>
<script>
const data = __DATA__;
const cell = data.cell;
const rows = data.maze.length, cols = data.maze[0].length;
const states = data.states, last = states.length - 1;
const STEPS_PER_SECOND = 4;

const canvas = document.getElementById("maze");
canvas.width = cols * cell;
canvas.height = rows * cell;
const ctx = canvas.getContext("2d");
const slider = document.getElementById("slider");
slider.max = last;

// Return after each step (for the info line).
const returns = [0];
for (const r of data.rewards) returns.push(returns[returns.length - 1] + r);

let t = 0;            // current time in steps (0 ... last), may be fractional
let playing = false;

function drawEmoji(emoji, x, y, size) {
  ctx.font = size + "px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(emoji, x, y);
}

function draw() {
  const i = Math.min(Math.floor(t), last);
  const f = t - i;
  const now = states[i], next = states[Math.min(i + 1, last)];
  const visited = new Set(now.visited.map(c => c[0] + "," + c[1]));

  // Tiles
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      let symbol = data.maze[y][x];
      if ((symbol === "k" && now.has_key) || (symbol === "D" && now.door_open)) symbol = ".";
      const tile = data.tiles[symbol];
      ctx.fillStyle = (symbol === "." || symbol === "^" || symbol === "k") ? data.tiles["."].color : tile.color;
      ctx.fillRect(x * cell, y * cell, cell, cell);
      if (visited.has(x + "," + y)) {
        ctx.fillStyle = "rgba(90, 60, 30, 0.15)";   // footprints
        ctx.fillRect(x * cell, y * cell, cell, cell);
      }
      if (tile.emoji) drawEmoji(tile.emoji, (x + 0.5) * cell, (y + 0.5) * cell, cell * 0.7);
    }
  }

  // The cat moves smoothly from one cell to the next.
  const x = now.x + (next.x - now.x) * f;
  const y = now.y + (next.y - now.y) * f;
  // Turning is smooth too: interpolate the angle the short way round.
  let turn = next.direction - now.direction;
  if (turn === 3) turn = -1;
  if (turn === -3) turn = 1;
  const angle = (now.direction + turn * f) * Math.PI / 2;   // 0 = north
  const cx = (x + 0.5) * cell, cy = (y + 0.5) * cell;

  // A small triangle shows where the cat is looking.
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(angle);
  ctx.fillStyle = "rgba(230, 120, 30, 0.9)";
  ctx.beginPath();
  ctx.moveTo(0, -cell * 0.5);
  ctx.lineTo(-cell * 0.16, -cell * 0.32);
  ctx.lineTo(cell * 0.16, -cell * 0.32);
  ctx.fill();
  ctx.restore();
  drawEmoji(data.agent, cx, cy, cell * 0.7);

  const step = Math.round(t);
  document.getElementById("info").textContent =
    "step " + step + " / " + last +
    "   return " + returns[step].toFixed(2) +
    "   " + (states[step].has_key ? "🗝️" : "") +
    (step === last ? "   (end)" : "");
  slider.value = t;
}

let previous = null;
function tick(timestamp) {
  if (previous !== null && playing) {
    const speed = parseFloat(document.getElementById("speed").value);
    t = Math.min(t + (timestamp - previous) / 1000 * STEPS_PER_SECOND * speed, last);
    if (t >= last) { playing = false; document.getElementById("play").textContent = "▶"; }
    draw();
  }
  previous = timestamp;
  requestAnimationFrame(tick);
}

document.getElementById("play").onclick = function () {
  if (t >= last) t = 0;
  playing = !playing;
  this.textContent = playing ? "⏸" : "▶";
};
document.getElementById("restart").onclick = function () { t = 0; draw(); };
slider.oninput = function () { t = parseFloat(slider.value); draw(); };

// Legend, built from the tile table.
document.getElementById("legend").innerHTML =
  "<span>" + data.agent + " agent</span>" +
  Object.values(data.tiles).map(tile => "<span>" +
    (tile.emoji || "<span class='swatch' style='background:" + tile.color + "'></span>") +
    " " + tile.name + "</span>").join("");

draw();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""
