"""Mazes for the dojo: generation, validity checks and ASCII/JSON input/output.

A maze is a list of strings. Each string is one row, each character one cell:

    #  wall      .  floor     <  entry     >  exit
    k  key       D  door      ^  trap

Coordinates: x = column, y = row, origin in the top-left corner (the same as a
Godot TileMap). So the cell at (x, y) is ``maze[y][x]``.

The agent is not stored in the maze (its position belongs to the environment
state). It starts on the floor cell next to the entry, looking away from the
entry. The symbol "@" is only used when we print a maze together with the agent.
"""

import json
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# The tile table
# ---------------------------------------------------------------------------
# Every tile type is described here, and only here. Other code (observation
# channels in env.py, colors and legend in viewer.py) is derived from this
# table. To add a new tile type: add one row here, then add its rule in env.py.
#
#   symbol   character in the ASCII maze
#   name     name used in code and in the viewer legend
#   color    (R, G, B) color for simple rendering
#   emoji    symbol for the HTML replay player ("" = draw only the color)
#   channel  observation channel in which the agent sees this tile
#            (None = the agent sees an empty cell)
TILES = [
    {"symbol": "#", "name": "wall",  "color": (60, 64, 82),    "emoji": "",   "channel": "wall"},
    {"symbol": ".", "name": "floor", "color": (236, 232, 220), "emoji": "",   "channel": None},
    # The entry cannot be walked on (the agent cannot leave the maze this
    # way), so the agent sees it as a wall.
    {"symbol": "<", "name": "entry", "color": (150, 190, 230), "emoji": "",   "channel": "wall"},
    {"symbol": ">", "name": "exit",  "color": (120, 200, 120), "emoji": "🏁", "channel": "exit"},
    {"symbol": "k", "name": "key",   "color": (240, 200, 60),  "emoji": "🗝️", "channel": "key"},
    {"symbol": "D", "name": "door",  "color": (150, 95, 50),   "emoji": "🚪", "channel": "door"},
    {"symbol": "^", "name": "trap",  "color": (220, 80, 80),   "emoji": "🔺", "channel": "trap"},
]

# The agent is not a tile, but it is drawn like one.
AGENT = {"symbol": "@", "name": "agent", "color": (240, 140, 40), "emoji": "🐱"}

# Lookup tables derived from TILES.
TILE = {tile["symbol"]: tile for tile in TILES}            # symbol -> table row
SYMBOL = {tile["name"]: tile["symbol"] for tile in TILES}  # name -> symbol

WALL = SYMBOL["wall"]
FLOOR = SYMBOL["floor"]
ENTRY = SYMBOL["entry"]
EXIT = SYMBOL["exit"]
KEY = SYMBOL["key"]
DOOR = SYMBOL["door"]
TRAP = SYMBOL["trap"]

# Directions: 0 = north (up), 1 = east (right), 2 = south (down), 3 = west (left).
# DIRECTIONS[d] is the (dx, dy) step for direction d. Note that y grows downwards.
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

# The maze from the seminar web page.
DEMO_MAZE = [
    "###########",
    "<...#.....#",
    "#.#.#.#.#.#",
    "#.#...#.#.#",
    "#.#####.#.#",
    "#...^...#.#",
    "###.#.#####",
    "#k..#....D>",
    "###########",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
# All functions accept a maze as a list of strings. The generator works on a
# list of lists of characters (so it can change cells); the helpers work on
# both, because maze[y][x] means the same for both.

def inside(maze, x: int, y: int) -> bool:
    """True if (x, y) is a cell of the maze."""
    return 0 <= y < len(maze) and 0 <= x < len(maze[0])


def find(maze, symbol: str) -> list[tuple[int, int]]:
    """All cells (x, y) that contain `symbol`, row by row."""
    return [(x, y)
            for y, row in enumerate(maze)
            for x, cell in enumerate(row)
            if cell == symbol]


def start_of(maze) -> tuple[int, int, int]:
    """Return (x, y, direction) of the agent at the start.

    The start is the open cell next to the entry; the agent looks away from
    the entry, into the maze.
    """
    entries = find(maze, ENTRY)
    if not entries:
        raise ValueError("the maze has no entry '<'")
    entry_x, entry_y = entries[0]
    for direction, (dx, dy) in enumerate(DIRECTIONS):
        x, y = entry_x + dx, entry_y + dy
        if inside(maze, x, y) and maze[y][x] != WALL:
            return x, y, direction
    raise ValueError("the entry has no open cell next to it")


def reachable(maze, start: tuple[int, int], blocked: set[str]) -> set[tuple[int, int]]:
    """Breadth-first search: all cells reachable from `start`.

    The search never enters a cell whose symbol is in `blocked`.
    Example: reachable(maze, start, {WALL, ENTRY, DOOR}) = everything the agent
    can reach before it has the key.
    """
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if (inside(maze, nx, ny) and (nx, ny) not in seen
                    and maze[ny][nx] not in blocked):
                seen.add((nx, ny))
                queue.append((nx, ny))
    return seen


def open_neighbors(maze, x: int, y: int) -> int:
    """Number of neighbor cells that are not walls."""
    return sum(1 for dx, dy in DIRECTIONS
               if inside(maze, x + dx, y + dy) and maze[y + dy][x + dx] != WALL)


# ---------------------------------------------------------------------------
# Validity check
# ---------------------------------------------------------------------------

def check(maze) -> list[str]:
    """Check that a maze is valid and solvable.

    Returns a list of problems (as text). An empty list means the maze is OK.
    Rules:
      - all rows have the same length, only symbols from TILES are used
      - the border is wall, except exactly one entry and one exit
      - the start cell (next to the entry) is floor
      - the exit can be reached without stepping on a trap
      - key and door: at most one of each, and both or none. The key can be
        reached without passing the door, and the exit cannot be reached
        without passing the door.
    """
    # 1. Shape and symbols. If these are wrong, the other checks make no sense.
    if len(maze) == 0 or len(maze[0]) == 0:
        return ["the maze is empty"]
    height, width = len(maze), len(maze[0])
    if any(len(row) != width for row in maze):
        return ["the rows have different lengths"]
    unknown = {cell for row in maze for cell in row} - set(TILE)
    if unknown:
        return [f"unknown symbols: {sorted(unknown)}"]

    # 2. Border, entry, exit, key and door.
    problems = []
    for y in range(height):
        for x in range(width):
            on_border = x in (0, width - 1) or y in (0, height - 1)
            cell = maze[y][x]
            if on_border and cell not in (WALL, ENTRY, EXIT):
                problems.append(f"the border cell {(x, y)} must be a wall, entry or exit")
            if not on_border and cell in (ENTRY, EXIT):
                problems.append(f"the entry/exit {(x, y)} must be on the border")
    entries, exits = find(maze, ENTRY), find(maze, EXIT)
    keys, doors = find(maze, KEY), find(maze, DOOR)
    if len(entries) != 1:
        problems.append(f"need exactly one entry '<', found {len(entries)}")
    if len(exits) != 1:
        problems.append(f"need exactly one exit '>', found {len(exits)}")
    if len(keys) > 1 or len(doors) > 1:
        problems.append("at most one key and one door are allowed")
    if len(keys) != len(doors):
        problems.append("a key needs a door and a door needs a key")
    if problems:
        return problems

    # 3. The start cell.
    try:
        x, y, _ = start_of(maze)
    except ValueError as error:
        return [str(error)]
    if maze[y][x] != FLOOR:
        return [f"the start cell {(x, y)} must be floor"]
    start, exit_cell = (x, y), exits[0]

    # 4. Paths. A maze must be solvable without stepping on a trap.
    if doors:
        before_door = reachable(maze, start, {WALL, ENTRY, TRAP, DOOR})
        if keys[0] not in before_door:
            problems.append("the key cannot be reached without passing the door (or a trap)")
        # Even walking over traps, there must be no way around the door.
        if exit_cell in reachable(maze, start, {WALL, ENTRY, DOOR}):
            problems.append("the exit can be reached without passing the door")
    # With the key, the door opens; the exit must then be reachable.
    if exit_cell not in reachable(maze, start, {WALL, ENTRY, TRAP}):
        problems.append("the exit cannot be reached without stepping on a trap")
    return problems


# ---------------------------------------------------------------------------
# Maze generation
# ---------------------------------------------------------------------------

def generate(width: int, height: int, seed: int | None = None, loops: float = 0.0,
             key_door: bool = False, traps: int = 0) -> list[str]:
    """Generate a random maze.

    width, height  odd numbers >= 5 (walls sit on even rows/columns)
    seed           the same seed always gives the same maze
    loops          fraction (0..1) of the remaining inner walls that are removed.
                   0 = "perfect maze" (exactly one path between two cells),
                   larger values create cycles, so there are several paths.
    key_door       place a door that every path to the exit must pass, and a
                   key that can be reached before the door
    traps          number of traps. They never block all paths, so there may
                   be fewer traps than asked for if there is no room.

    The entry is on the left border and the exit on the right border (like
    the demo maze), each in a random row.
    """
    if width % 2 == 0 or height % 2 == 0 or width < 5 or height < 5:
        raise ValueError(f"width and height must be odd numbers >= 5, got {width} x {height}")
    if not 0.0 <= loops <= 1.0:
        raise ValueError(f"loops must be between 0 and 1, got {loops}")
    if traps < 0:
        raise ValueError(f"traps must be >= 0, got {traps}")

    rng = np.random.default_rng(seed)  # our own random generator, no global state
    grid = [[WALL] * width for _ in range(height)]
    _carve_perfect_maze(grid, rng)
    _add_loops(grid, rng, loops)
    _place_entry_and_exit(grid, rng)
    if key_door:
        _place_key_and_door(grid, rng)
    _place_traps(grid, rng, traps)

    maze = ["".join(row) for row in grid]
    problems = check(maze)
    if problems:  # this would be a bug in the generator
        raise RuntimeError(f"generated an invalid maze (seed {seed}): {problems}")
    return maze


def _carve_perfect_maze(grid, rng) -> None:
    """Depth-first search maze ("recursive backtracker").

    Cells with odd x and odd y are rooms; the cells between them are walls.
    We walk from room to room in random order and remove the wall between
    two rooms when we visit a new room. When we get stuck, we go back
    (backtrack) until there is a room we have not visited yet. The result
    has exactly one path between any two rooms.
    """
    height, width = len(grid), len(grid[0])
    grid[1][1] = FLOOR
    stack = [(1, 1)]
    while stack:
        x, y = stack[-1]
        # Rooms two cells away that are still untouched (= all wall).
        new_rooms = [(x + 2 * dx, y + 2 * dy) for dx, dy in DIRECTIONS
                     if 0 < x + 2 * dx < width - 1 and 0 < y + 2 * dy < height - 1
                     and grid[y + 2 * dy][x + 2 * dx] == WALL]
        if not new_rooms:
            stack.pop()  # stuck: go back
            continue
        nx, ny = new_rooms[rng.integers(len(new_rooms))]
        grid[(y + ny) // 2][(x + nx) // 2] = FLOOR  # remove the wall in between
        grid[ny][nx] = FLOOR
        stack.append((nx, ny))


def _add_loops(grid, rng, loops: float) -> None:
    """Remove a fraction of the inner walls between two rooms (creates cycles)."""
    height, width = len(grid), len(grid[0])
    # A wall between two rooms has exactly one odd coordinate. (Walls with two
    # even coordinates are "pillars"; removing them would make open areas.)
    walls = [(x, y)
             for y in range(1, height - 1)
             for x in range(1, width - 1)
             if grid[y][x] == WALL and (x + y) % 2 == 1]
    n_remove = round(loops * len(walls))
    for i in rng.choice(len(walls), size=n_remove, replace=False):
        x, y = walls[i]
        grid[y][x] = FLOOR


def _place_entry_and_exit(grid, rng) -> None:
    """Entry on the left border, exit on the right border, in random odd rows
    (odd rows are next to a room, so there is always floor behind them)."""
    height, width = len(grid), len(grid[0])
    odd_rows = list(range(1, height - 1, 2))
    grid[odd_rows[rng.integers(len(odd_rows))]][0] = ENTRY
    grid[odd_rows[rng.integers(len(odd_rows))]][width - 1] = EXIT


def _is_corridor(grid, x: int, y: int) -> bool:
    """True if the cell has walls on both sides (left+right or up+down),
    so a door there looks like a door."""
    return ((grid[y][x - 1] == WALL and grid[y][x + 1] == WALL)
            or (grid[y - 1][x] == WALL and grid[y + 1][x] == WALL))


def _place_key_and_door(grid, rng) -> None:
    """Put the door on a cell that every path to the exit must pass (a
    "bottleneck"), and the key somewhere the agent can reach before the door."""
    start_x, start_y, _ = start_of(grid)
    start = (start_x, start_y)
    exit_cell = find(grid, EXIT)[0]

    # Try a door on every floor cell and keep the cells where it works:
    # the exit must be unreachable, and there must be room for the key.
    options = []  # list of (door cell, possible key cells)
    for x, y in find(grid, FLOOR):
        if (x, y) == start:
            continue
        grid[y][x] = DOOR  # try a door here ...
        before_door = reachable(grid, start, {WALL, ENTRY, DOOR})
        grid[y][x] = FLOOR  # ... and remove it again
        if exit_cell in before_door:
            continue  # the agent could walk around this door
        key_cells = sorted(cell for cell in before_door if cell != start)
        if key_cells:
            options.append(((x, y), key_cells))

    # Prefer doors in corridors, they look better.
    corridor_options = [option for option in options if _is_corridor(grid, *option[0])]
    if corridor_options:
        options = corridor_options
    (door_x, door_y), key_cells = options[rng.integers(len(options))]

    # Prefer dead ends for the key, so the agent has to search a little.
    dead_ends = [(x, y) for x, y in key_cells if open_neighbors(grid, x, y) == 1]
    if dead_ends:
        key_cells = dead_ends
    key_x, key_y = key_cells[rng.integers(len(key_cells))]

    grid[door_y][door_x] = DOOR
    grid[key_y][key_x] = KEY


def _place_traps(grid, rng, n_traps: int) -> None:
    """Put up to `n_traps` traps on random floor cells (not the start).
    A trap is only kept if the maze stays solvable without stepping on a trap."""
    start_x, start_y, _ = start_of(grid)
    candidates = [cell for cell in find(grid, FLOOR) if cell != (start_x, start_y)]
    placed = 0
    for i in rng.permutation(len(candidates)):
        if placed == n_traps:
            break
        x, y = candidates[i]
        grid[y][x] = TRAP
        if check(grid):          # problems: this trap blocks the way
            grid[y][x] = FLOOR   # so take it away again
        else:
            placed += 1


# ---------------------------------------------------------------------------
# ASCII and JSON input/output
# ---------------------------------------------------------------------------

def to_ascii(maze, agent: tuple[int, int] | None = None) -> str:
    """The maze as text, one row per line. If `agent` = (x, y) is given,
    the agent is shown as '@'."""
    rows = [list(row) for row in maze]
    if agent is not None:
        x, y = agent[0], agent[1]
        rows[y][x] = AGENT["symbol"]
    return "\n".join("".join(row) for row in rows)


def from_ascii(text: str) -> list[str]:
    """Read a maze from text (one row per line). Empty lines and spaces at the
    start/end of a line are ignored."""
    maze = [line.strip() for line in text.splitlines() if line.strip()]
    if not maze or any(len(row) != len(maze[0]) for row in maze):
        raise ValueError("a maze needs at least one row and all rows must have the same length")
    return maze


def to_json(maze) -> str:
    """The maze as JSON text: {"width": W, "height": H, "rows": [...]}.
    This is the format for Godot and for saved mazes."""
    data = {"width": len(maze[0]), "height": len(maze), "rows": ["".join(row) for row in maze]}
    return json.dumps(data, indent=2)


def from_json(text: str) -> list[str]:
    """Read a maze from JSON text written by to_json()."""
    data = json.loads(text)
    maze = list(data["rows"])
    if len(maze) != data["height"] or any(len(row) != data["width"] for row in maze):
        raise ValueError("width/height do not match the rows")
    return maze
