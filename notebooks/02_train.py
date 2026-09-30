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
# # 02 — Training an agent with PPO (弟子の修行)
#
# We train a neural network policy with **PPO** (Proximal Policy Optimization).
# The whole algorithm is in `dojo/train.py` (about 150 lines). Read it from
# top to bottom!
#
# 1. **Rollout**: 16 cats walk through 16 mazes for 128 steps each.
# 2. **Advantages (GAE)**: for each step: "was this action better or worse than expected?"
# 3. **Update**: make good actions more likely and bad actions less likely,
#    but not too much at once (clipping).
# 4. Repeat.

# %% [markdown]
# ## Setup (セットアップ)

# %%
import sys

if "google.colab" in sys.modules:
    # TODO (step 6): install from the real repository URL here, e.g.
    #   pip install -q git+https://github.com/USER/dojo
    pass

# %%
from dojo import maze as mz
from dojo.agents import wall_follower
from dojo.env import REWARDS, MazeEnv
from dojo.train import (Config, actor_policy, evaluate_with_baselines, export_weights,
                        load_checkpoint, load_weights, print_results, train, weights_policy)
from dojo.viewer import record_episode, replay_html, save_gif

# %% [markdown]
# The rewards (報酬). You can change them and train again — what happens?

# %%
REWARDS

# %% [markdown]
# ## 1. Small perfect mazes (小さい迷路)
# 300,000 steps take about 20–30 seconds on a laptop CPU.
# Watch `success` go up and `length` go down.

# %%
config = Config(width=9, height=9, steps=300_000)
actor, critic, curve = train(config)

# %% [markdown]
# Evaluation on 50 **new** mazes (never seen during training), next to the
# baselines. "greedy" = always take the most likely action; "sampling" =
# choose randomly with the learned probabilities.

# %%
results = evaluate_with_baselines(actor, config)
print_results(results)

# %% [markdown]
# Let's watch the trained cat on a new maze. 🐱

# %%
env = MazeEnv(width=9, height=9)
episode = record_episode(env, actor_policy(actor, greedy=False), seed=1_000_123)
print("solved:", episode["solved"], " steps:", len(episode["actions"]))
replay_html(episode)

# %% [markdown]
# ## 2. Harder mazes: key and door (鍵と扉)
# The same training from the command line. This command also works in a
# terminal or in a Slurm job script. 1,000,000 steps take about 1–2 minutes.

# %%
# !{sys.executable} -m dojo.train --steps 1000000 --key-door --out runs/keydoor9

# %% [markdown]
# The run folder contains plain files:
# `checkpoint.pt`, `weights.json`, `curve.csv`, `eval.json`.

# %%
actor, critic, config = load_checkpoint("runs/keydoor9/checkpoint.pt")
env = MazeEnv(width=9, height=9, key_door=True)
episode = record_episode(env, actor_policy(actor, greedy=False), seed=1_000_007)
print("solved:", episode["solved"], " steps:", len(episode["actions"]))
replay_html(episode)

# %%
save_gif(episode, "runs/keydoor9/replay.gif")

# %% [markdown]
# **Try it (やってみよう):**
# - `--loops 0.1` (cycles: the wall follower can fail)
# - `--width 11 --height 11 --loops 0.1 --key-door --traps 3` (everything)
# - `--maze demo` (always the same maze — why does this fail? Hint: where is
#   the key, and where does the compass point?)
# - `--maze demo --explored` (the agent also sees how much of the maze it has
#   explored — now it works. Why? Does it also help on new mazes?)
# - Change `REWARDS["new_cell"]` to 0 and train again.

# %% [markdown]
# ## 3. Export for the arena (アリーナ用のエクスポート)
# `weights.json` contains the actor weights **and** the env config. The arena
# (Godot) computes the same forward pass as this small NumPy function.

# %%
weights = load_weights("runs/keydoor9/weights.json", env.config)
print("config:", weights["config"])
print("layers:", [(len(layer["W"]), len(layer["W"][0])) for layer in weights["layers"]])

obs, _ = env.reset(seed=1_000_007)
print("NumPy action:", weights_policy(weights, greedy=True)(obs),
      " PyTorch action:", actor_policy(actor, greedy=True)(obs))

# %% [markdown]
# ## 4. Optional: Training with Stable Baselines3 (任意)
# Stable Baselines3 (SB3) is a well-documented RL library. Our `MazeEnv` is a
# standard Gymnasium env, so SB3 works without changes.
# Install with `pip install "dojo[sb3]"`. This takes about 3 minutes.

# %%
try:
    import stable_baselines3  # noqa: F401
    has_sb3 = True
except ImportError:
    has_sb3 = False
    print("SB3 is not installed: skipping this section")

# %%
if has_sb3:
    from stable_baselines3 import PPO

    from dojo.train import evaluate, export_weights_sb3, sb3_policy

    env = MazeEnv(width=9, height=9)
    model = PPO("MlpPolicy", env, seed=0)
    model.learn(300_000)

    print("SB3 PPO      :", evaluate(sb3_policy(model), env, n_mazes=50))
    print("wall follower:", evaluate(wall_follower(env), env, n_mazes=50))
    # Same weights JSON format -> SB3 agents can play in the same arena.
    export_weights_sb3(model, "runs/sb3_weights.json")
