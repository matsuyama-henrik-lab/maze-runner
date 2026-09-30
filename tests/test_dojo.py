"""Tests for the dojo. Run with:  pytest -q"""

import json

import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env
from PIL import Image

from dojo import maze as mz
from dojo.agents import random_agent, wall_follower
from dojo.env import CHANNELS, FORWARD, LEFT, REWARDS, MazeEnv, obs_layout
from dojo.train import (Config, check_config, evaluate, export_weights, forward, load_weights,
                        make_actor, softmax, train, weights_policy)
from dojo.viewer import grid_at, record_episode, replay_page, save_gif

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


# ---------------------------------------------------------------------------
# env.py and agents.py
# ---------------------------------------------------------------------------


# An open room: good for testing what the agent sees.
ROOM = mz.from_ascii("""
    #######
    #.....#
    #.....#
    <..^..#
    #.....#
    #.....#
    #####>#
""")

# A corridor with key and door: start (1,1), key (2,1), door (4,1).
CORRIDOR = mz.from_ascii("""
    ########
    <.k.D..>
    ########
""")


def run_episode(env, policy, seed=None):
    obs, info = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += reward
        done = terminated or truncated
    return total, terminated, info


def view_channel(env, obs, name):
    view = env.config["view"]
    return obs[:view * view * len(CHANNELS)].reshape(len(CHANNELS), view, view)[CHANNELS.index(name)]


@pytest.mark.filterwarnings("ignore:.*alternative render modes")
def test_check_env():
    check_env(MazeEnv(), skip_render_check=False)
    check_env(MazeEnv(maze=mz.DEMO_MAZE, key_door=True))


def test_obs_layout():
    assert CHANNELS == ["wall", "key", "door", "trap", "exit", "visited"]
    env = MazeEnv()
    obs, _ = env.reset(seed=0)
    assert env.layout == {"view": (0, 150), "has_key": (150, 151), "compass": (151, 153)}
    assert obs.shape == (153,) and obs.dtype == np.float32
    assert obs_layout({"view": 3, "compass": False, "explored": False}) == \
        {"view": (0, 54), "has_key": (54, 55)}
    assert MazeEnv(view=3, compass=False).observation_space.shape == (55,)
    assert MazeEnv(explored=True).layout["explored"] == (153, 154)


def test_explored_fraction():
    env = MazeEnv(maze=CORRIDOR, explored=True)   # 7 walkable cells (with the exit)
    obs, _ = env.reset()
    assert obs[-1] == pytest.approx(1 / 7)
    env.step(FORWARD)
    obs, *_ = env.step(FORWARD)
    assert obs[-1] == pytest.approx(3 / 7)
    obs, *_ = env.step(LEFT)                      # turning does not visit new cells
    assert obs[-1] == pytest.approx(3 / 7)


def test_reset_seed_selects_maze():
    env = MazeEnv(width=11, height=9, key_door=True, traps=2)
    env.reset(seed=42)
    assert env.maze == mz.generate(11, 9, seed=42, key_door=True, traps=2)


def test_wall_ahead_is_in_the_same_slot_for_every_direction():
    env = MazeEnv(maze=ROOM)
    env.reset()
    c = env.config["view"] // 2
    # Positions next to each outer wall, facing that wall.
    for (x, y, direction) in [(3, 1, 0), (5, 3, 1), (2, 5, 2), (1, 2, 3)]:
        env.x, env.y, env.direction = x, y, direction
        walls = view_channel(env, env._observation(), "wall")
        assert walls[c - 1, c] == 1       # wall directly ahead
        assert walls[c, c - 1] == 0 and walls[c, c + 1] == 0  # open left and right
        assert walls[c, c] == 0           # the agent's own cell


def test_view_rotates_with_the_agent():
    # Facing east, the world is rotated counterclockwise by 90 degrees, etc.
    env = MazeEnv(width=11, height=11, loops=0.2, key_door=True, traps=3)
    env.reset(seed=1)
    for (x, y) in [(3, 3), (5, 5), (1, 7)]:
        env.x, env.y, env.direction = x, y, 0
        north = env._view_window()
        for direction in range(4):
            env.direction = direction
            assert np.array_equal(env._view_window(), np.rot90(north, k=direction, axes=(1, 2)))


def test_trap_in_view_and_compass():
    env = MazeEnv(maze=ROOM)
    env.reset()
    env.x, env.y, env.direction = 1, 3, 1          # trap two cells ahead
    c = env.config["view"] // 2
    traps = view_channel(env, env._observation(), "trap")
    assert traps[c - 2, c] == 1 and traps.sum() == 1
    obs = env._observation()
    forward, right = obs[env.layout["compass"][0]:]
    assert forward > 0 and right > 0                # exit (5,6) is ahead-right


def test_door_blocks_without_key_and_opens_with_key():
    env = MazeEnv(maze=CORRIDOR)
    env.reset()
    env.x = 3                                       # skip the key, stand before the door
    _, reward, *_ = env.step(FORWARD)
    assert (env.x, env.door_open) == (3, False)     # blocked
    assert reward == pytest.approx(REWARDS["step"])
    env.has_key = True
    _, reward, *_ = env.step(FORWARD)
    assert (env.x, env.door_open) == (4, True)
    assert reward == pytest.approx(REWARDS["step"] + REWARDS["door"] + REWARDS["new_cell"])
    assert env.grid[1][4] == mz.FLOOR


def test_key_door_exit_rewards():
    env = MazeEnv(maze=CORRIDOR)
    obs, _ = env.reset()
    assert obs[env.layout["has_key"][0]] == 0
    rewards = []
    for _ in range(6):
        obs, reward, terminated, truncated, _ = env.step(FORWARD)
        rewards.append(reward)
    assert env.has_key and obs[env.layout["has_key"][0]] == 1
    assert terminated and not truncated
    assert sum(rewards) == pytest.approx(6 * REWARDS["step"] + 6 * REWARDS["new_cell"]
                                         + REWARDS["key"] + REWARDS["door"] + REWARDS["exit"])


def test_trap_costs_but_episode_continues():
    env = MazeEnv(maze=mz.DEMO_MAZE)
    env.reset()
    env.x, env.y, env.direction = 3, 5, 1           # trap (4,5) ahead
    _, reward, terminated, _, _ = env.step(FORWARD)
    assert (env.x, env.y) == (4, 5) and not terminated
    assert reward == pytest.approx(REWARDS["step"] + REWARDS["trap"] + REWARDS["new_cell"])
    assert env.state()["trap_hits"] == 1
    # Stepping back and onto the trap again: no new-cell bonus, but the trap hurts again.
    env.step(LEFT), env.step(LEFT), env.step(FORWARD), env.step(LEFT), env.step(LEFT)
    _, reward, *_ = env.step(FORWARD)
    assert reward == pytest.approx(REWARDS["step"] + REWARDS["trap"])
    assert env.state()["trap_hits"] == 2


def test_truncation_and_state():
    env = MazeEnv(maze=mz.DEMO_MAZE, max_steps=10)
    env.reset()
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(LEFT)
    assert truncated and not terminated
    state = env.state()
    assert json.loads(json.dumps(state)) == state
    assert state["steps"] == 10 and state["visited"] == [[1, 1]]


def test_render():
    env = MazeEnv(maze=mz.DEMO_MAZE, render_mode="rgb_array")
    env.reset()
    assert env.render().shape == (9 * 16, 11 * 16, 3)
    env = MazeEnv(maze=mz.DEMO_MAZE, render_mode="ansi")
    env.reset()
    assert env.render().splitlines()[1] == "<@..#.....#"


@pytest.mark.parametrize("key_door", [False, True])
def test_wall_follower_solves_perfect_mazes(key_door):
    for width, height in [(5, 5), (9, 9), (15, 11)]:
        env = MazeEnv(width=width, height=height, key_door=key_door)
        policy = wall_follower(env)
        for seed in range(15):
            _, terminated, info = run_episode(env, policy, seed=seed)
            assert terminated and info["at_exit"], (width, height, seed)


def test_random_agent_runs():
    env = MazeEnv(width=7, height=7)
    _, _, info = run_episode(env, random_agent, seed=0)
    assert info["steps"] > 0


# ---------------------------------------------------------------------------
# viewer.py
# ---------------------------------------------------------------------------


def test_record_episode_and_replay(tmp_path):
    env = MazeEnv(maze=mz.DEMO_MAZE)
    episode = record_episode(env, wall_follower(env))
    assert episode["solved"]
    assert len(episode["states"]) == len(episode["actions"]) + 1 == len(episode["rewards"]) + 1
    assert json.loads(json.dumps(episode)) == episode
    page = replay_page(episode)
    assert page.startswith("<!DOCTYPE html>") and "__DATA__" not in page
    save_gif(episode, tmp_path / "demo.gif")
    assert Image.open(tmp_path / "demo.gif").n_frames == len(episode["states"])


def test_grid_at_removes_key_and_opened_door():
    state = {"has_key": True, "door_open": True}
    grid = grid_at(CORRIDOR, state)
    assert "".join(grid[1]) == "<......>"


# ---------------------------------------------------------------------------
# train.py
# ---------------------------------------------------------------------------

TINY = mz.from_ascii("""
    #######
    <.....#
    #####.#
    #.....#
    #.#####
    #.....>
    #######
""")


def test_numpy_forward_matches_torch_actor(tmp_path):
    torch.manual_seed(0)
    env = MazeEnv(width=11, height=11, key_door=True, traps=2)
    actor = make_actor(env.observation_space.shape[0], env.action_space.n)
    path = tmp_path / "weights.json"
    export_weights(actor, path, env.config)
    weights = load_weights(path, env.config)
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        for _ in range(10):
            logits = actor(torch.as_tensor(obs)).detach().numpy()
            assert np.allclose(forward(weights, obs), logits, atol=1e-5)
            assert weights_policy(weights, greedy=True)(obs) == int(np.argmax(logits))
            assert np.allclose(softmax(forward(weights, obs)), torch.softmax(torch.as_tensor(logits), 0).numpy(), atol=1e-6)
            obs, *_ = env.step(int(np.argmax(logits)))


def test_weights_with_wrong_config_are_rejected(tmp_path):
    env = MazeEnv()
    path = tmp_path / "weights.json"
    export_weights(make_actor(153, 3), path, env.config)
    with pytest.raises(ValueError):
        load_weights(path, MazeEnv(view=3).config)
    check_config(load_weights(path), env.config)   # the right config is fine


def test_ppo_improves_on_tiny_maze(tmp_path):
    maze_file = tmp_path / "tiny.txt"
    maze_file.write_text(mz.to_ascii(TINY))
    config = Config(maze=str(maze_file), steps=50_000, n_envs=4, n_rollout=64, seed=0)
    actor, critic, curve = train(config, verbose=False)
    returns = [row["mean_return"] for row in curve if row["mean_return"] is not None]
    assert returns[-1] > returns[0] + 0.5
    # The trained agent solves the maze, also through the exported weights.
    env = MazeEnv(maze=TINY)
    export_weights(actor, tmp_path / "weights.json", env.config)
    policy = weights_policy(load_weights(tmp_path / "weights.json", env.config), greedy=True)
    assert evaluate(policy, env, n_mazes=1)["success_rate"] == 1


def test_sb3_export_matches_sb3_model(tmp_path):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from dojo.train import export_weights_sb3, sb3_policy

    env = MazeEnv(width=7, height=7)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, seed=0)
    model.learn(128)
    export_weights_sb3(model, tmp_path / "sb3.json")
    weights = load_weights(tmp_path / "sb3.json", env.config)
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        for _ in range(10):
            action = sb3_policy(model)(obs)
            assert weights_policy(weights, greedy=True)(obs) == action
            obs, *_ = env.step(action)


def test_weights_policy_samples_with_softmax(tmp_path):
    env = MazeEnv()
    path = tmp_path / "weights.json"
    export_weights(make_actor(153, 3), path, env.config)   # untrained: ~uniform
    policy = weights_policy(load_weights(path), seed=0)
    obs, _ = env.reset(seed=0)
    counts = np.bincount([policy(obs) for _ in range(600)], minlength=3)
    assert all(counts > 150)   # all three actions are chosen
