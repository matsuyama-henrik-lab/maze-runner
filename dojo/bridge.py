"""Bridge to the Godot arena: a small TCP server that plays the game.

Python owns the game rules (MazeEnv); Godot only displays the game and (later)
lets humans play. They talk over a local TCP connection:

    python -m dojo.bridge --weights runs/demo/weights.json   # a trained agent
    python -m dojo.bridge                                     # the wall follower

Protocol: one JSON message per line (UTF-8, ending with "\\n").

    Godot -> Python                        Python -> Godot
    {"type": "reset", "seed": 3}           {"type": "reset", "maze": {...}, "state": {...},
                                            "config": {...}, "obs_layout": {...}}
    {"type": "step"}                       {"type": "step", "action": a, "state": {...},
      (the agent chooses the action)        "reward": r, "done": b}
    {"type": "step", "action": a}          (same reply)
      (a human chose the action)
    anything wrong                         {"type": "error", "message": "..."}

"seed" is optional (without it, a random maze). With --maze the maze is fixed.
See godot/README.md for details.
"""

import argparse
import json
import socket

from dojo import maze as mz
from dojo.agents import wall_follower
from dojo.env import ACTIONS, MazeEnv
from dojo.train import check_config, load_maze, load_weights, weights_policy

HOST = "127.0.0.1"   # only this computer can connect
PORT = 11008


def new_game(env: MazeEnv, policy) -> dict:
    """Everything the server needs to remember between messages."""
    return {"env": env, "policy": policy, "obs": None, "done": True}


def handle_message(game: dict, message: dict) -> dict:
    """Answer one message from Godot. This function does not know about
    sockets, so it is easy to test."""
    env = game["env"]
    kind = message.get("type")

    if kind == "reset":
        game["obs"], _ = env.reset(seed=message.get("seed"))
        game["done"] = False
        return {"type": "reset", "maze": json.loads(mz.to_json(env.maze)),
                "state": env.state(), "config": env.config, "obs_layout": env.layout}

    if kind == "step":
        if game["obs"] is None:
            return {"type": "error", "message": "send a reset first"}
        if game["done"]:
            return {"type": "error", "message": "the episode is over, send a reset"}
        if "action" in message:                    # a human chose the action
            action = message["action"]
            if action not in range(len(ACTIONS)):
                return {"type": "error", "message": f"invalid action {action!r}"}
        else:                                      # the agent chooses
            action = game["policy"](game["obs"])
        game["obs"], reward, terminated, truncated, _ = env.step(action)
        game["done"] = terminated or truncated
        return {"type": "step", "action": int(action), "state": env.state(),
                "reward": float(reward), "done": game["done"]}

    return {"type": "error", "message": f"unknown message type {kind!r}"}


def make_server(host: str = HOST, port: int = PORT) -> socket.socket:
    """A listening TCP socket. port=0 lets the system choose a free port (for tests)."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # restart without waiting
    server.bind((host, port))
    server.listen(1)
    return server


def serve(server: socket.socket, env: MazeEnv, policy, max_clients: int | None = None) -> None:
    """Serve clients one after another (Godot connects, plays, disconnects).
    max_clients=None: forever."""
    served = 0
    while max_clients is None or served < max_clients:
        connection, address = server.accept()
        # Send small messages immediately instead of collecting them first
        # (otherwise every step can be delayed by ~40 ms).
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"client connected: {address}", flush=True)
        game = new_game(env, policy)
        with connection, connection.makefile("r", encoding="utf-8") as lines:
            for line in lines:                     # one JSON message per line
                if not line.strip():
                    continue
                try:
                    reply = handle_message(game, json.loads(line))
                except json.JSONDecodeError as error:
                    reply = {"type": "error", "message": f"invalid JSON: {error}"}
                connection.sendall((json.dumps(reply) + "\n").encode("utf-8"))
        print("client disconnected", flush=True)
        served += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the maze game to the Godot arena.")
    parser.add_argument("--weights", help="weights.json of a trained agent (default: wall follower)")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--width", type=int, default=9)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--loops", type=float, default=0.0)
    parser.add_argument("--key-door", action="store_true")
    parser.add_argument("--traps", type=int, default=0)
    parser.add_argument("--maze", default=None, help='"demo" or a maze file: always this maze')
    args = parser.parse_args()

    maze = load_maze(args.maze) if args.maze else None
    settings = dict(maze=maze, width=args.width, height=args.height, loops=args.loops,
                    key_door=args.key_door, traps=args.traps)
    if args.weights:
        weights = load_weights(args.weights)
        # The env must produce exactly the observation the agent was trained on.
        config = weights["config"]
        env = MazeEnv(view=config["view"], compass=config["compass"],
                      explored=config["explored"], **settings)
        check_config(weights, env.config)          # raises if the config does not match
        policy = weights_policy(weights)           # softmax sampling
        print(f"agent: {args.weights}", flush=True)
    else:
        env = MazeEnv(**settings)
        policy = wall_follower(env)
        print("agent: wall follower", flush=True)

    server = make_server(port=args.port)
    print(f"listening on {HOST}:{args.port} (Ctrl+C to stop)", flush=True)
    try:
        serve(server, env, policy)
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        server.close()


if __name__ == "__main__":
    main()
