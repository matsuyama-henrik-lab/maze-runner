"""Baseline agents. Every learned agent is compared against these.

A *policy* is a function  policy(obs) -> action.
"""

import numpy as np

from dojo import maze as mz
from dojo.env import FORWARD, LEFT, RIGHT, MazeEnv

_rng = np.random.default_rng()


def random_agent(obs: np.ndarray) -> int:
    """Choose one of the three actions at random (ignores the observation)."""
    return int(_rng.integers(3))


def wall_follower(env: MazeEnv):
    """Right-hand rule: keep your right hand on the wall and you will find the
    exit of every maze without cycles.

    The wall follower "cheats" a little: it reads the maze from the env state
    instead of the observation. It returns a policy for this env:

        policy = wall_follower(env)
        action = policy(obs)
    """
    just_turned_right = False  # remembered between calls

    def policy(obs) -> int:
        nonlocal just_turned_right
        if env.steps == 0:  # a new episode has started
            just_turned_right = False

        def open_towards(direction):
            dx, dy = mz.DIRECTIONS[direction % 4]
            return env.is_open(env.x + dx, env.y + dy)

        # After a right turn we must walk forward; otherwise we could turn
        # right forever in an open area.
        if just_turned_right and open_towards(env.direction):
            action = FORWARD
        elif open_towards(env.direction + 1):   # free on the right: turn right
            action = RIGHT
        elif open_towards(env.direction):       # free ahead: go forward
            action = FORWARD
        else:                                   # blocked: turn left
            action = LEFT
        just_turned_right = action == RIGHT
        return action

    return policy
