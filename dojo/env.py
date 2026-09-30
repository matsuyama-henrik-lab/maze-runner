"""MazeEnv: a Gymnasium environment in which an agent walks through a maze.

The agent (a cat) starts next to the entry, may pick up a key, open a door,
should avoid traps and must reach the exit.

Actions (egocentric = relative to where the agent looks):
    0 = forward, 1 = turn left, 2 = turn right

Observation: one flat float32 vector, built from named parts in this order
(obs_layout() returns the exact index ranges):

    "view"     view x view cells around the agent, rotated so that the agent
               always looks "up" in the window. The agent is in the center
               cell. One-hot channels (see CHANNELS):
                   wall, key, door, trap, exit, visited
               Cells outside the maze count as wall.
               Index: channel * view * view + row * view + col
                   row 0 = farthest ahead, row view // 2 = agent's row
                   col 0 = far left,       col view // 2 = agent's column
               So the cell directly ahead is (row, col) = (view//2 - 1, view//2).
    "has_key"  1.0 if the agent carries the key, else 0.0
    "compass"  (only if compass=True) unit vector from the agent to the exit,
               in the agent's frame: (forward, right). (0, 0) at the exit.
    "explored" (only if explored=True) fraction of the walkable cells the agent
               has visited (0..1). A kind of "clock": on a fixed maze it lets
               the agent tell "at the start" and "back at the start later"
               apart. It does not help much on new mazes.

With the defaults (view=5, compass=True, explored=False): 5*5*6 + 1 + 2 = 153 values.
Godot must build exactly the same vector, so change this layout only
together with OBS_VERSION.
"""

import gymnasium as gym
import numpy as np

from dojo import maze as mz

# ---------------------------------------------------------------------------
# Rewards: change these to change what the agent learns.
# ---------------------------------------------------------------------------
REWARDS = {
    "step": -0.01,   # every step costs a little, so short paths are better
    "new_cell": 0.02,  # first visit of a cell: rewards exploring (0 = switch off)
    "key": 0.2,      # picking up the key
    "door": 0.2,     # opening the door (with the key)
    "trap": -0.5,    # stepping on a trap (the episode continues!)
    "exit": 1.0,     # reaching the exit (the episode ends)
}

ACTIONS = ["forward", "turn left", "turn right"]
FORWARD, LEFT, RIGHT = 0, 1, 2

# Observation channels of the view window: one per tile channel in the tile
# table (in table order, without duplicates), plus "visited".
CHANNELS = list(dict.fromkeys(t["channel"] for t in mz.TILES if t["channel"])) + ["visited"]

# Increase this whenever the observation layout changes, so old agents are
# not used with a new layout by mistake.
OBS_VERSION = 1

# Maze seeds used when reset() is called without a seed (= training) are
# below this number. Evaluation uses seeds >= this number, so the test
# mazes are never seen during training.
TRAIN_SEED_LIMIT = 1_000_000


def obs_layout(config: dict) -> dict[str, tuple[int, int]]:
    """Names and index ranges [start, end) of the observation parts.

    Example (defaults): {"view": (0, 150), "has_key": (150, 151), "compass": (151, 153)}
    Optional parts (compass, explored) are only present when switched on.
    """
    sizes = {"view": config["view"] ** 2 * len(CHANNELS), "has_key": 1}
    if config["compass"]:
        sizes["compass"] = 2
    if config["explored"]:
        sizes["explored"] = 1
    layout, start = {}, 0
    for name, size in sizes.items():
        layout[name] = (start, start + size)
        start += size
    return layout


class MazeEnv(gym.Env):
    """A maze environment.

    maze        a fixed maze (list of strings), e.g. mz.DEMO_MAZE. If None, every
                reset() generates a new maze with the settings below.
    width, height, loops, key_door, traps
                settings for mz.generate()
    view        size of the view window (odd number)
    compass     add the direction to the exit to the observation
    explored    add the fraction of visited cells to the observation
    max_steps   the episode is truncated after this many steps
                (default: 4 x number of cells of the maze)
    render_mode "rgb_array" or "ansi"
    """

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 8}

    def __init__(self, maze: list[str] | None = None, width: int = 9, height: int = 9,
                 loops: float = 0.0, key_door: bool = False, traps: int = 0,
                 view: int = 5, compass: bool = True, explored: bool = False,
                 max_steps: int | None = None,
                 render_mode: str | None = None):
        if maze is not None:
            problems = mz.check(maze)
            if problems:
                raise ValueError(f"invalid maze: {problems}")
        if view % 2 == 0 or view < 3:
            raise ValueError(f"view must be an odd number >= 3, got {view}")
        self.fixed_maze = maze
        self.generator_settings = dict(width=width, height=height, loops=loops,
                                       key_door=key_door, traps=traps)
        self.max_steps_setting = max_steps
        self.render_mode = render_mode

        # The configuration travels with a trained agent (weights JSON, bridge),
        # because the agent only works with the observation it was trained on.
        self.config = {"view": view, "compass": compass, "explored": explored,
                       "obs_version": OBS_VERSION}
        self.layout = obs_layout(self.config)
        n_obs = max(end for _, end in self.layout.values())
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(n_obs,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))

    # -----------------------------------------------------------------------
    # reset and step
    # -----------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Start a new episode.

        With a fixed maze, the maze is always the same. Otherwise reset(seed=s)
        uses maze number s (the same as mz.generate(..., seed=s)), and reset()
        without a seed picks a random training maze.
        """
        super().reset(seed=seed)  # sets up self.np_random
        if self.fixed_maze is not None:
            self.maze = list(self.fixed_maze)
            self.maze_seed = None
        else:
            if seed is None:
                seed = int(self.np_random.integers(TRAIN_SEED_LIMIT))
            self.maze = mz.generate(seed=seed, **self.generator_settings)
            self.maze_seed = seed

        # The grid can change during an episode (key picked up, door opened).
        self.grid = [list(row) for row in self.maze]
        self.x, self.y, self.direction = mz.start_of(self.maze)
        self.exit_x, self.exit_y = mz.find(self.maze, mz.EXIT)[0]
        self.n_walkable = sum(1 for row in self.maze for tile in row
                              if tile not in (mz.WALL, mz.ENTRY))
        self.has_key = False
        self.door_open = False
        self.visited = {(self.x, self.y)}
        self.steps = 0
        self.trap_hits = 0
        self.max_steps = self.max_steps_setting or 4 * len(self.maze) * len(self.maze[0])
        return self._observation(), self._info()

    def step(self, action: int):
        action = int(action)
        if action not in (FORWARD, LEFT, RIGHT):
            raise ValueError(f"invalid action {action}")
        reward = REWARDS["step"]
        terminated = False

        if action == LEFT:
            self.direction = (self.direction + 3) % 4
        elif action == RIGHT:
            self.direction = (self.direction + 1) % 4
        else:  # FORWARD
            dx, dy = mz.DIRECTIONS[self.direction]
            x, y = self.x + dx, self.y + dy
            can_enter, tile_reward, terminated = self._enter_tile(x, y)
            reward += tile_reward
            if can_enter:
                self.x, self.y = x, y
                if (x, y) not in self.visited:
                    reward += REWARDS["new_cell"]
                    self.visited.add((x, y))

        self.steps += 1
        truncated = self.steps >= self.max_steps and not terminated
        return self._observation(), reward, terminated, truncated, self._info()

    # -----------------------------------------------------------------------
    # Tile rules: what happens when the agent walks into a cell.
    # A new tile type gets its rule here (and its row in mz.TILES).
    # -----------------------------------------------------------------------

    def _enter_tile(self, x: int, y: int) -> tuple[bool, float, bool]:
        """The agent tries to walk into cell (x, y).

        Returns (can_enter, reward, episode_ends). Tiles may change here
        (the key disappears, the door opens).
        """
        if not mz.inside(self.grid, x, y):
            return False, 0.0, False
        tile = self.grid[y][x]

        if tile in (mz.WALL, mz.ENTRY):
            return False, 0.0, False            # blocked
        if tile == mz.DOOR:
            if not self.has_key:
                return False, 0.0, False        # closed door = like a wall
            self.grid[y][x] = mz.FLOOR          # the key opens the door
            self.door_open = True
            return True, REWARDS["door"], False
        if tile == mz.KEY:
            self.grid[y][x] = mz.FLOOR          # pick up the key
            self.has_key = True
            return True, REWARDS["key"], False
        if tile == mz.TRAP:
            self.trap_hits += 1                 # it hurts, but we can walk on
            return True, REWARDS["trap"], False
        if tile == mz.EXIT:
            return True, REWARDS["exit"], True  # done!
        return True, 0.0, False                 # floor

    def is_open(self, x: int, y: int) -> bool:
        """True if the agent could walk into (x, y) right now
        (used by the wall follower). Traps count as open."""
        if not mz.inside(self.grid, x, y):
            return False
        tile = self.grid[y][x]
        if tile == mz.DOOR:
            return self.has_key
        return tile not in (mz.WALL, mz.ENTRY)

    # -----------------------------------------------------------------------
    # Observation
    # -----------------------------------------------------------------------

    def _observation(self) -> np.ndarray:
        parts = {"view": self._view_window().ravel(),
                 "has_key": [float(self.has_key)]}
        if self.config["compass"]:
            parts["compass"] = self._compass()
        if self.config["explored"]:
            parts["explored"] = [len(self.visited) / self.n_walkable]
        # Concatenate in the order of obs_layout().
        return np.concatenate([np.asarray(parts[name], dtype=np.float32)
                               for name in self.layout])

    def _view_window(self) -> np.ndarray:
        """One-hot view window, shape (channels, view, view), rotated so the
        agent looks up (row 0 = ahead)."""
        view = self.config["view"]
        center = view // 2
        window = np.zeros((len(CHANNELS), view, view), dtype=np.float32)
        forward = mz.DIRECTIONS[self.direction]
        right = mz.DIRECTIONS[(self.direction + 1) % 4]
        for row in range(view):
            for col in range(view):
                ahead = center - row     # cells ahead of the agent (negative = behind)
                side = col - center      # cells to the right (negative = left)
                x = self.x + ahead * forward[0] + side * right[0]
                y = self.y + ahead * forward[1] + side * right[1]
                if not mz.inside(self.grid, x, y):
                    window[CHANNELS.index("wall"), row, col] = 1.0
                    continue
                channel = mz.TILE[self.grid[y][x]]["channel"]
                if channel is not None:
                    window[CHANNELS.index(channel), row, col] = 1.0
                if (x, y) in self.visited:
                    window[CHANNELS.index("visited"), row, col] = 1.0
        return window

    def _compass(self) -> list[float]:
        """Unit vector to the exit in the agent's frame: (forward, right)."""
        dx, dy = self.exit_x - self.x, self.exit_y - self.y
        forward = mz.DIRECTIONS[self.direction]
        right = mz.DIRECTIONS[(self.direction + 1) % 4]
        f = dx * forward[0] + dy * forward[1]
        r = dx * right[0] + dy * right[1]
        length = (f * f + r * r) ** 0.5
        return [f / length, r / length] if length > 0 else [0.0, 0.0]

    # -----------------------------------------------------------------------
    # State, info and rendering
    # -----------------------------------------------------------------------

    def state(self) -> dict:
        """Small JSON-able description of the current state (for the replay
        viewer and the Godot bridge)."""
        return {
            "x": self.x,
            "y": self.y,
            "direction": self.direction,
            "has_key": self.has_key,
            "door_open": self.door_open,
            "visited": sorted([x, y] for x, y in self.visited),
            "steps": self.steps,
            "trap_hits": self.trap_hits,
        }

    def _info(self) -> dict:
        return {"maze_seed": self.maze_seed, "steps": self.steps,
                "trap_hits": self.trap_hits, "has_key": self.has_key,
                "at_exit": (self.x, self.y) == (self.exit_x, self.exit_y)}

    def render(self):
        if self.render_mode == "ansi":
            return mz.to_ascii(self.grid, agent=(self.x, self.y))
        if self.render_mode == "rgb_array":
            return draw(self.grid, self.state())
        return None


def draw(grid, state: dict, cell_size: int = 16) -> np.ndarray:
    """Draw the maze as an RGB image (height, width, 3), one colored block
    per cell. Visited cells are a bit darker; the agent is a square with a
    small mark on the side it is looking at."""
    height, width = len(grid), len(grid[0])
    image = np.zeros((height * cell_size, width * cell_size, 3), dtype=np.uint8)
    visited = {tuple(cell) for cell in state["visited"]}

    def fill(x, y, color, margin=0):
        top, left = y * cell_size + margin, x * cell_size + margin
        size = cell_size - 2 * margin
        image[top:top + size, left:left + size] = color

    for y in range(height):
        for x in range(width):
            color = np.array(mz.TILE[grid[y][x]]["color"])
            if (x, y) in visited and grid[y][x] == mz.FLOOR:
                color = color * 0.85
            fill(x, y, color)

    # The agent and a "nose" in its viewing direction.
    x, y = state["x"], state["y"]
    fill(x, y, mz.AGENT["color"], margin=cell_size // 6)
    dx, dy = mz.DIRECTIONS[state["direction"]]
    nose = max(cell_size // 5, 1)
    cx = x * cell_size + cell_size // 2 + dx * (cell_size // 3) - nose // 2
    cy = y * cell_size + cell_size // 2 + dy * (cell_size // 3) - nose // 2
    image[cy:cy + nose, cx:cx + nose] = (40, 30, 30)
    return image
