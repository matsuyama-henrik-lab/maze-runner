"""Train an agent with PPO (Proximal Policy Optimization), in plain PyTorch.

Usage (terminal, notebook cell with "!", or an sbatch script):

    python -m dojo.train --steps 300000 --n-envs 16 --out runs/demo [--key-door] [--loops 0.1]

The whole learning algorithm is in this file, top to bottom:
    1. collect a rollout: every env plays n_rollout steps with the current policy
    2. compute advantages with GAE ("how much better than expected was this action?")
    3. update actor and critic for a few epochs on minibatches:
       clipped policy loss + value loss - entropy bonus
    4. repeat until `steps` environment steps are done

Outputs in --out (plain files):
    checkpoint.pt   actor + critic weights and the config (torch.save)
    weights.json    actor weights for the NumPy / GDScript forward pass
    curve.csv       training curve (one row per update)
    eval.json       evaluation on held-out mazes, next to the baselines
"""

import argparse
import csv
import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from dojo import maze as mz
from dojo.agents import random_agent, wall_follower
from dojo.env import ACTIONS, TRAIN_SEED_LIMIT, MazeEnv, obs_layout


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Mazes (see MazeEnv / maze.generate)
    width: int = 9
    height: int = 9
    loops: float = 0.0
    key_door: bool = False
    traps: int = 0
    maze: str | None = None     # "demo" or a maze file (.json / .txt) = always the same maze
    view: int = 5
    compass: bool = True
    explored: bool = False      # add "fraction of cells visited" to the observation
    # PPO
    steps: int = 300_000        # total environment steps (all envs together)
    n_envs: int = 16            # environments played in parallel
    n_rollout: int = 128        # steps per env before each update
    n_epochs: int = 4           # passes over the rollout per update
    n_minibatches: int = 4      # minibatches per epoch
    lr: float = 3e-4            # learning rate (Adam)
    gamma: float = 0.99         # discount factor: how much future rewards count
    gae_lambda: float = 0.95    # GAE: trade-off between bias and variance
    clip: float = 0.2           # PPO clipping range for the probability ratio
    ent_coef: float = 0.01      # entropy bonus: keeps the policy exploring
    vf_coef: float = 0.5        # weight of the value loss
    max_grad_norm: float = 0.5  # gradient clipping
    # Other
    seed: int = 0
    device: str = "cpu"
    out: str = "runs/demo"
    eval_mazes: int = 50


def make_env(config: Config) -> MazeEnv:
    """A MazeEnv with the maze settings from the config."""
    maze = load_maze(config.maze) if config.maze else None
    return MazeEnv(maze=maze, width=config.width, height=config.height, loops=config.loops,
                   key_door=config.key_door, traps=config.traps,
                   view=config.view, compass=config.compass, explored=config.explored)


def load_maze(name: str) -> list[str]:
    """"demo" = the demo maze, otherwise a .json or ASCII text file."""
    if name == "demo":
        return mz.DEMO_MAZE
    text = Path(name).read_text()
    return mz.from_json(text) if name.endswith(".json") else mz.from_ascii(text)


# ---------------------------------------------------------------------------
# Networks: actor (policy) and critic (value function), two separate MLPs
# ---------------------------------------------------------------------------

def layer(n_in: int, n_out: int, std: float = 2 ** 0.5) -> nn.Linear:
    """A linear layer with orthogonal initialization (standard for PPO; it
    makes training more stable than the PyTorch default)."""
    linear = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(linear.weight, std)
    nn.init.zeros_(linear.bias)
    return linear


def make_actor(n_obs: int, n_actions: int) -> nn.Sequential:
    """Observation -> action logits (unnormalized log-probabilities)."""
    # The small std of the last layer makes all actions nearly equally
    # likely at the start, so the agent explores.
    return nn.Sequential(layer(n_obs, 64), nn.Tanh(), layer(64, 64), nn.Tanh(),
                         layer(64, n_actions, std=0.01))


def make_critic(n_obs: int) -> nn.Sequential:
    """Observation -> value (expected future reward from here)."""
    return nn.Sequential(layer(n_obs, 64), nn.Tanh(), layer(64, 64), nn.Tanh(),
                         layer(64, 1, std=1.0))


# ---------------------------------------------------------------------------
# PPO training
# ---------------------------------------------------------------------------

def train(config: Config, verbose: bool = True) -> tuple[nn.Sequential, nn.Sequential, list[dict]]:
    """Train actor and critic with PPO. Returns (actor, critic, curve).

    `curve` is a list of dicts, one per update (also written to curve.csv).
    """
    torch.manual_seed(config.seed)
    # Our networks are tiny: one CPU thread is faster than many (less overhead).
    torch.set_num_threads(1)
    device = torch.device(config.device)
    envs = [make_env(config) for _ in range(config.n_envs)]
    n_obs = envs[0].observation_space.shape[0]
    n_actions = envs[0].action_space.n

    actor = make_actor(n_obs, n_actions).to(device)
    critic = make_critic(n_obs).to(device)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()),
                                 lr=config.lr, eps=1e-5)

    # Rollout buffers, shape (n_rollout, n_envs, ...)
    T, N = config.n_rollout, config.n_envs
    buf_obs = torch.zeros((T, N, n_obs), device=device)
    buf_actions = torch.zeros((T, N), dtype=torch.long, device=device)
    buf_logprobs = torch.zeros((T, N), device=device)
    buf_rewards = torch.zeros((T, N), device=device)
    buf_dones = torch.zeros((T, N), device=device)   # 1 = the episode ended after this step
    buf_values = torch.zeros((T, N), device=device)

    # Start all envs. Each env gets a different first maze; later mazes come
    # from the env's own random generator (always a training seed).
    obs = np.stack([env.reset(seed=config.seed * N + i)[0] for i, env in enumerate(envs)])
    episode_return = np.zeros(N)
    recent = deque(maxlen=100)    # (return, solved, length) of the last 100 episodes
    curve = []
    n_updates = max(1, config.steps // (T * N))
    batch_size = T * N
    minibatch_size = batch_size // config.n_minibatches
    start_time = time.time()

    for update in range(1, n_updates + 1):
        # Linearly decrease the learning rate to 0 (makes the end of training calmer).
        optimizer.param_groups[0]["lr"] = config.lr * (1 - (update - 1) / n_updates)

        # --- 1. Collect a rollout with the current policy -------------------
        for t in range(T):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                dist = Categorical(logits=actor(obs_tensor))
                action = dist.sample()
                value = critic(obs_tensor).squeeze(1)
            buf_obs[t] = obs_tensor
            buf_actions[t] = action
            buf_logprobs[t] = dist.log_prob(action)
            buf_values[t] = value

            for i, env in enumerate(envs):
                next_obs, reward, terminated, truncated, info = env.step(action[i].item())
                episode_return[i] += reward
                if truncated and not terminated:
                    # The episode was cut off by the time limit, not really
                    # finished: add the value of where we are (bootstrapping).
                    with torch.no_grad():
                        final = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                        reward += config.gamma * critic(final).item()
                if terminated or truncated:
                    recent.append((episode_return[i], terminated, info["steps"]))
                    episode_return[i] = 0.0
                    next_obs, _ = env.reset()
                buf_rewards[t, i] = reward
                buf_dones[t, i] = float(terminated or truncated)
                obs[i] = next_obs

        # --- 2. Advantages with GAE (Generalized Advantage Estimation) -------
        # delta_t = r_t + gamma * V(s_t+1) - V(s_t): was step t better than expected?
        # advantage_t = delta_t + gamma * lambda * advantage_t+1 (a smoothed sum of deltas)
        with torch.no_grad():
            next_value = critic(torch.as_tensor(obs, dtype=torch.float32, device=device)).squeeze(1)
            advantages = torch.zeros((T, N), device=device)
            last_advantage = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                not_done = 1.0 - buf_dones[t]   # no future after an episode ended
                value_after = next_value if t == T - 1 else buf_values[t + 1]
                delta = buf_rewards[t] + config.gamma * value_after * not_done - buf_values[t]
                last_advantage = delta + config.gamma * config.gae_lambda * not_done * last_advantage
                advantages[t] = last_advantage
            returns = advantages + buf_values   # targets for the critic

        # --- 3. PPO update on minibatches ------------------------------------
        b_obs = buf_obs.reshape(batch_size, n_obs)
        b_actions = buf_actions.reshape(batch_size)
        b_logprobs = buf_logprobs.reshape(batch_size)
        b_advantages = advantages.reshape(batch_size)
        b_returns = returns.reshape(batch_size)

        for epoch in range(config.n_epochs):
            order = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                idx = order[start:start + minibatch_size]
                dist = Categorical(logits=actor(b_obs[idx]))
                new_logprob = dist.log_prob(b_actions[idx])
                entropy = dist.entropy().mean()
                new_value = critic(b_obs[idx]).squeeze(1)

                # Normalize advantages in the minibatch (mean 0, std 1).
                adv = b_advantages[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Clipped policy loss: ratio = new probability / old probability.
                # Clipping stops the policy from changing too much in one update.
                ratio = torch.exp(new_logprob - b_logprobs[idx])
                policy_loss = -torch.min(ratio * adv,
                                         torch.clamp(ratio, 1 - config.clip, 1 + config.clip) * adv).mean()
                # Value loss: the critic should predict the returns.
                value_loss = 0.5 * ((new_value - b_returns[idx]) ** 2).mean()
                # Total: minimize policy and value loss, maximize entropy (exploration).
                loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()),
                                         config.max_grad_norm)
                optimizer.step()

        # --- 4. Log progress ---------------------------------------------------
        with torch.no_grad():
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()  # how much the policy changed
        row = {
            "step": update * batch_size,
            "time_s": round(time.time() - start_time, 1),
            "episodes": len(recent),
            "mean_return": round(float(np.mean([r[0] for r in recent])), 3) if recent else None,
            "success_rate": round(float(np.mean([r[1] for r in recent])), 3) if recent else None,
            "mean_length": round(float(np.mean([r[2] for r in recent])), 1) if recent else None,
            "policy_loss": round(policy_loss.item(), 4),
            "value_loss": round(value_loss.item(), 4),
            "entropy": round(entropy.item(), 4),
            "approx_kl": round(approx_kl, 5),
        }
        curve.append(row)
        if verbose and (update % 10 == 0 or update == n_updates):
            print(f"step {row['step']:>8}  {row['time_s']:>6.1f}s  return {row['mean_return']}"
                  f"  success {row['success_rate']}  length {row['mean_length']}"
                  f"  entropy {row['entropy']:.3f}", flush=True)  # flush: show it now (e.g. in Slurm logs)
    return actor, critic, curve


def load_checkpoint(path: str) -> tuple[nn.Sequential, nn.Sequential, Config]:
    """Load actor, critic and training config from a checkpoint.pt."""
    checkpoint = torch.load(path, map_location="cpu")
    config = Config(**checkpoint["config"])
    env = make_env(config)
    actor = make_actor(env.observation_space.shape[0], env.action_space.n)
    critic = make_critic(env.observation_space.shape[0])
    actor.load_state_dict(checkpoint["actor"])
    critic.load_state_dict(checkpoint["critic"])
    return actor, critic, config


# ---------------------------------------------------------------------------
# Policies and evaluation
# ---------------------------------------------------------------------------

def actor_policy(actor: nn.Sequential, greedy: bool = True, device: str = "cpu"):
    """Turn the actor network into a policy function obs -> action.
    greedy=True: always take the most likely action (like the arena does)."""
    def policy(obs) -> int:
        with torch.no_grad():
            logits = actor(torch.as_tensor(obs, dtype=torch.float32, device=device))
        if greedy:
            return int(torch.argmax(logits).item())
        return int(Categorical(logits=logits).sample().item())
    return policy


def evaluate(policy, env: MazeEnv, n_mazes: int = 50, first_seed: int = TRAIN_SEED_LIMIT) -> dict:
    """Play one episode on each of `n_mazes` held-out mazes (seeds from
    `first_seed` on, never used in training). For a fixed maze, the same
    maze is played `n_mazes` times.

    Returns success rate, mean steps, mean trap hits and mean return.
    """
    solved, steps, trap_hits, returns = [], [], [], []
    for seed in range(first_seed, first_seed + n_mazes):
        obs, info = env.reset(seed=seed)
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total += reward
            done = terminated or truncated
        solved.append(terminated)
        steps.append(info["steps"])
        trap_hits.append(info["trap_hits"])
        returns.append(total)
    return {"success_rate": float(np.mean(solved)), "mean_steps": float(np.mean(steps)),
            "mean_trap_hits": float(np.mean(trap_hits)), "mean_return": float(np.mean(returns))}


def evaluate_with_baselines(actor: nn.Sequential, config: Config) -> dict:
    """Evaluate the trained actor (greedy and sampling) next to the baselines."""
    env = make_env(config)
    return {
        "ppo (greedy)": evaluate(actor_policy(actor, True, config.device), env, config.eval_mazes),
        "ppo (sampling)": evaluate(actor_policy(actor, False, config.device), env, config.eval_mazes),
        "wall follower": evaluate(wall_follower(env), env, config.eval_mazes),
        "random": evaluate(random_agent, env, config.eval_mazes),
    }


def print_results(results: dict) -> None:
    print(f"{'agent':16s} {'success':>8s} {'steps':>7s} {'traps':>6s} {'return':>7s}")
    for name, r in results.items():
        print(f"{name:16s} {r['success_rate']:8.0%} {r['mean_steps']:7.1f} "
              f"{r['mean_trap_hits']:6.2f} {r['mean_return']:7.2f}")


# ---------------------------------------------------------------------------
# Weight export (for the Godot arena) and the NumPy forward pass
# ---------------------------------------------------------------------------
# Weights JSON format:
#   {"config": {"view": 5, "compass": true, "explored": false, "obs_version": 1},
#    "obs_layout": {"view": [0, 150], ...}, "actions": [...], "activation": "tanh",
#    "layers": [{"W": [[...], ...], "b": [...]}, ...]}
# W has shape (n_out, n_in) (like torch.nn.Linear), so a layer computes W @ x + b.
# tanh after every layer except the last one; the last layer gives the logits.

def _weights_dict(layers: list[tuple[np.ndarray, np.ndarray]], env_config: dict) -> dict:
    return {
        "config": dict(env_config),
        "obs_layout": obs_layout(env_config),
        "actions": ACTIONS,
        "activation": "tanh",
        "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in layers],
    }


def export_weights(actor: nn.Sequential, path: str, env_config: dict) -> None:
    """Save the actor MLP weights (and the env config) as JSON."""
    layers = [(m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy())
              for m in actor if isinstance(m, nn.Linear)]
    Path(path).write_text(json.dumps(_weights_dict(layers, env_config)))


def load_weights(path: str, env_config: dict | None = None) -> dict:
    """Load a weights JSON. If `env_config` is given, it must match the
    config the agent was trained with (otherwise the observation is different)."""
    weights = json.loads(Path(path).read_text())
    if env_config is not None:
        check_config(weights, env_config)
    return weights


def check_config(weights: dict, env_config: dict) -> None:
    if weights["config"] != dict(env_config):
        raise ValueError(f"this agent was trained with env config {weights['config']}, "
                         f"but the env uses {dict(env_config)}")


def forward(weights: dict, obs: np.ndarray) -> np.ndarray:
    """The actor in plain NumPy: observation -> logits.
    This is exactly what the GDScript version in Godot has to compute."""
    x = np.asarray(obs, dtype=np.float64)
    layers = weights["layers"]
    for i, layer_weights in enumerate(layers):
        x = np.asarray(layer_weights["W"]) @ x + np.asarray(layer_weights["b"])
        if i < len(layers) - 1:
            x = np.tanh(x)
    return x


def softmax(logits: np.ndarray) -> np.ndarray:
    """Logits -> probabilities (subtracting the max avoids overflow in exp)."""
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def weights_policy(weights: dict, greedy: bool = False, seed: int | None = None):
    """Policy function from a weights JSON.

    greedy=False (default, used in the arena): choose the action randomly with
    the softmax probabilities. A greedy agent always does the same thing in
    the same situation and can get stuck in a loop.
    greedy=True: always the action with the largest logit.
    """
    rng = np.random.default_rng(seed)

    def policy(obs) -> int:
        logits = forward(weights, obs)
        if greedy:
            return int(np.argmax(logits))
        return int(rng.choice(len(logits), p=softmax(logits)))
    return policy


# ---------------------------------------------------------------------------
# Optional: Stable Baselines3 (pip install dojo[sb3])
# ---------------------------------------------------------------------------

def export_weights_sb3(model, path: str, env_config: dict | None = None) -> None:
    """Export the actor of an SB3 PPO "MlpPolicy" to the same weights JSON
    format, so SB3-trained agents can play in the same arena.

    Works for the default network (two hidden layers, tanh), which is the
    same shape as ours. If `env_config` is None, it is read from the model's env.
    """
    policy = model.policy
    if not isinstance(policy.activation_fn(), nn.Tanh):
        raise ValueError("only policies with tanh activation can be exported")
    if env_config is None:
        env_config = model.get_env().get_attr("config")[0]
    # SB3's actor = mlp_extractor.policy_net (hidden layers) + action_net (logits).
    linears = [m for m in policy.mlp_extractor.policy_net if isinstance(m, nn.Linear)]
    linears.append(policy.action_net)
    layers = [(m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy()) for m in linears]
    Path(path).write_text(json.dumps(_weights_dict(layers, env_config)))


def sb3_policy(model):
    """Policy function from an SB3 model (greedy)."""
    return lambda obs: int(model.predict(obs, deterministic=True)[0])


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Train a maze agent with PPO.")
    parser.add_argument("--steps", type=int, default=defaults.steps, help="total environment steps")
    parser.add_argument("--n-envs", type=int, default=defaults.n_envs, help="parallel environments")
    parser.add_argument("--out", default=defaults.out, help="output folder")
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--height", type=int, default=defaults.height)
    parser.add_argument("--loops", type=float, default=defaults.loops)
    parser.add_argument("--key-door", action="store_true")
    parser.add_argument("--traps", type=int, default=defaults.traps)
    parser.add_argument("--explored", action="store_true",
                        help="add the fraction of visited cells to the observation")
    parser.add_argument("--maze", default=None, help='"demo" or a maze file: always train on this maze')
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--ent-coef", type=float, default=defaults.ent_coef)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device, help='"cpu" (default) or e.g. "cuda"')
    args = parser.parse_args()
    return Config(**vars(args))


def main() -> None:
    config = parse_args()
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"training on {config.device}, output in {out}/", flush=True)

    actor, critic, curve = train(config)
    env_config = make_env(config).config

    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "config": asdict(config), "env_config": env_config}, out / "checkpoint.pt")
    export_weights(actor, out / "weights.json", env_config)
    with open(out / "curve.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)

    print(f"\nevaluation on {config.eval_mazes} held-out mazes:")
    results = evaluate_with_baselines(actor, config)
    print_results(results)
    summary = {"config": asdict(config), "train_time_s": curve[-1]["time_s"], "results": results}
    (out / "eval.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
