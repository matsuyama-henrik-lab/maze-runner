"""Tests for the dojo. Run with:  pytest -q"""

import pytest

from dojo import maze as mz

# ---------------------------------------------------------------------------
# maze.py
# ---------------------------------------------------------------------------


def count_tiles(maze, symbol):
    return sum(row.count(symbol) for row in maze)


def floor_graph(maze):
    """Number of walkable cells and number of connections between them
    (entry excluded, because the agent cannot walk on it)."""
    cells = [(x, y) for y, row in enumerate(maze) for x, c in enumerate(row)
             if c not in (mz.WALL, mz.ENTRY)]
    cell_set = set(cells)
    # Count each connection once: only look right and down.
    edges = sum(1 for x, y in cells for nx, ny in ((x + 1, y), (x, y + 1)) if (nx, ny) in cell_set)
    return len(cells), edges


def test_demo_maze_is_valid():
    assert mz.check(mz.DEMO_MAZE) == []
    assert len(mz.DEMO_MAZE) == 9 and len(mz.DEMO_MAZE[0]) == 11
    assert mz.start_of(mz.DEMO_MAZE) == (1, 1, 1)  # next to the entry, facing east


def test_tile_table_is_consistent():
    symbols = [tile["symbol"] for tile in mz.TILES]
    assert len(symbols) == len(set(symbols))
    assert mz.AGENT["symbol"] not in symbols


@pytest.mark.parametrize("width,height", [(5, 5), (7, 9), (11, 11), (15, 11)])
@pytest.mark.parametrize("loops", [0.0, 0.2])
@pytest.mark.parametrize("key_door", [False, True])
@pytest.mark.parametrize("traps", [0, 3])
def test_generated_mazes_pass_check(width, height, loops, key_door, traps):
    for seed in range(25):
        maze = mz.generate(width, height, seed=seed, loops=loops, key_door=key_door, traps=traps)
        assert mz.check(maze) == [], (seed, mz.to_ascii(maze))
        assert len(maze) == height and len(maze[0]) == width
        assert count_tiles(maze, mz.KEY) == count_tiles(maze, mz.DOOR) == int(key_door)
        assert count_tiles(maze, mz.TRAP) <= traps


def test_generate_is_reproducible():
    assert mz.generate(11, 9, seed=7, key_door=True, traps=2) == \
        mz.generate(11, 9, seed=7, key_door=True, traps=2)
    assert mz.generate(11, 9, seed=7) != mz.generate(11, 9, seed=8)


def test_perfect_maze_has_no_cycles():
    # A maze without cycles is a tree: connections = cells - 1.
    for seed in range(20):
        cells, edges = floor_graph(mz.generate(11, 11, seed=seed))
        assert edges == cells - 1


def test_loops_create_cycles():
    for seed in range(20):
        cells, edges = floor_graph(mz.generate(11, 11, seed=seed, loops=0.3))
        assert edges > cells - 1


def test_traps_are_placed_when_there_is_room():
    for seed in range(20):
        maze = mz.generate(15, 15, seed=seed, loops=0.2, traps=3)
        assert count_tiles(maze, mz.TRAP) == 3


def test_invalid_arguments():
    with pytest.raises(ValueError):
        mz.generate(10, 9)   # even width
    with pytest.raises(ValueError):
        mz.generate(3, 9)    # too small
    with pytest.raises(ValueError):
        mz.generate(9, 9, loops=1.5)


def test_check_finds_problems():
    # Exit walled off.
    blocked = mz.from_ascii("""
        #######
        <.....#
        #####.#
        #...#.#
        #.###.#
        #....##
        ######>
    """)
    assert mz.check(blocked) != []
    # Key behind the door.
    key_behind_door = mz.from_ascii("""
        #######
        <...D.#
        #####k#
        #.....>
        #######
    """)
    assert mz.check(key_behind_door) != []
    # A door that can be bypassed.
    bypass = mz.from_ascii("""
        #######
        <.kD..#
        #.#####
        #.....>
        #######
    """)
    assert any("without passing the door" in p for p in mz.check(bypass))
    # The only path goes over a trap.
    trap_blocks = mz.from_ascii("""
        #####
        <.^.>
        #####
    """)
    assert mz.check(trap_blocks) != []
    # Rows of different length / unknown symbols.
    assert mz.check(["###", "<.>", "##"]) != []
    assert mz.check(["#####", "<.x.>", "#####"]) != []


def test_ascii_and_json_round_trip():
    maze = mz.generate(9, 7, seed=3, key_door=True, traps=1)
    assert mz.from_ascii(mz.to_ascii(maze)) == maze
    assert mz.from_json(mz.to_json(maze)) == maze
    shown = mz.to_ascii(mz.DEMO_MAZE, agent=(1, 1))
    assert shown.splitlines()[1] == "<@..#.....#"
