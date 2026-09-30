# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 — Mazes and the viewer (迷路とビューア)
#
# In this notebook we
# 1. look at mazes (迷路を見る),
# 2. walk through a maze step by step (一歩ずつ歩く),
# 3. watch two simple agents: a random agent and a wall follower (ランダム vs 右手法),
# 4. make a replay and a GIF.

# %% [markdown]
# ## Setup (セットアップ)
# On Google Colab, install the package first. Locally or on the server it is
# already installed.

# %%
import sys

if "google.colab" in sys.modules:
    # TODO (step 6): install from the real repository URL here, e.g.
    #   pip install -q git+https://github.com/USER/dojo
    pass

# %%
from dojo import maze as mz
from dojo.agents import random_agent, wall_follower
from dojo.env import TRAIN_SEED_LIMIT, MazeEnv
from dojo.viewer import LiveViewer, record_episode, replay_html, save_gif

# %% [markdown]
# ## 1. Mazes (迷路)
# A maze is just a list of strings: `#` wall, `.` floor, `<` entry, `>` exit,
# `k` key, `D` door, `^` trap. This is the demo maze from the seminar page:

# %%
print(mz.to_ascii(mz.DEMO_MAZE))

# %% [markdown]
# `generate()` makes new mazes. The same `seed` always gives the same maze.
# Try other sizes, `loops`, `key_door` and `traps`!

# %%
print(mz.to_ascii(mz.generate(15, 9, seed=1)))
print()
print(mz.to_ascii(mz.generate(15, 9, seed=1, loops=0.2, key_door=True, traps=3)))

# %% [markdown]
# `check()` tells us if a maze is valid. An empty list means OK.

# %%
print(mz.check(mz.DEMO_MAZE))
print(mz.check(["#####", "<.^.>", "#####"]))  # the only path goes over a trap

# %% [markdown]
# ## 2. The environment (環境)
# The agent can do three things: `0` = forward, `1` = turn left, `2` = turn right.
# After each action it gets an observation and a reward.

# %%
env = MazeEnv(maze=mz.DEMO_MAZE, render_mode="ansi")
obs, info = env.reset()
print(env.render())
print("observation size:", obs.shape, " parts:", env.layout)

# %%
for action in [0, 0, 2, 0, 0]:   # forward, forward, turn right, forward, forward
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"action {action}  reward {reward:+.2f}")
print(env.render())

# %% [markdown]
# ## 3. Live viewer (ライブビューア)
# The wall follower keeps its right hand on the wall. 🐱

# %%
env = MazeEnv(maze=mz.DEMO_MAZE, render_mode="rgb_array")
viewer = LiveViewer()
viewer.play(env, wall_follower(env), fps=8)

# %% [markdown]
# And a random agent on a generated maze (it stops after `max_steps`):

# %%
env = MazeEnv(width=9, height=9)
viewer = LiveViewer()
viewer.play(env, random_agent, seed=5, fps=30)

# %% [markdown]
# ## 4. Replay and GIF (リプレイとGIF)
# `record_episode` records everything; `replay_html` plays it back with
# play/pause, speed and a slider.

# %%
env = MazeEnv(maze=mz.DEMO_MAZE)
episode = record_episode(env, wall_follower(env))
print("solved:", episode["solved"], " steps:", len(episode["actions"]),
      " return:", round(sum(episode["rewards"]), 2))
replay_html(episode)

# %%
save_gif(episode, "wall_follower_demo.gif")
print("saved wall_follower_demo.gif")

# %% [markdown]
# ## 5. Random agent vs wall follower (比較)
# 50 new mazes (seeds that are never used for training). Which agent is
# better? Can a learned agent beat the wall follower? (→ notebook 02)

# %%
def success_rate(env, policy, n_mazes=50):
    solved = 0
    for seed in range(TRAIN_SEED_LIMIT, TRAIN_SEED_LIMIT + n_mazes):
        solved += record_episode(env, policy, seed=seed)["solved"]
    return solved / n_mazes


env = MazeEnv(width=9, height=9, key_door=True)
print("random agent :", success_rate(env, random_agent))
print("wall follower:", success_rate(env, wall_follower(env)))
