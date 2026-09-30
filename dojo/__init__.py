"""Dojo: train reinforcement-learning agents (弟子) to solve grid mazes.

Modules:
    maze     maze generation, checks, ASCII/JSON input/output
    env      the Gymnasium environment (MazeEnv)
    agents   baseline agents (random, wall follower)
    viewer   rendering, notebook viewer, HTML replay, GIF export
    train    our own PPO implementation, evaluation, weight export
    bridge   socket server for the Godot arena
"""

__version__ = "0.1.0"
