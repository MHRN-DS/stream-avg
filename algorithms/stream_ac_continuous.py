import pickle
import argparse
import json
from pathlib import Path
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from optim import ObGD as Optimizer
from sparse_init import sparse_init
from env_factory import make_train_env
from paths import ensure_run_dirs, train_csv_path, eval_csv_path, returns_pkl_path
from evaluation.fixed_evaluator import evaluate_policy
from logging_utils.csv_logger import CSVLogger


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

def save_run_config(results_root, backend, algo, env_name, seed, config):
    run_path = ensure_run_dirs(results_root, backend, algo, env_name, seed)
    config_path = Path(run_path) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent = 2, sort_keys = True)

class Actor(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128):
        super(Actor, self).__init__()
        self.fc_layer = nn.Linear(n_obs, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_mu = nn.Linear(hidden_size, n_actions)
        self.linear_std = nn.Linear(hidden_size, n_actions)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        mu = self.linear_mu(x)
        pre_std = self.linear_std(x)
        std = F.softplus(pre_std)
        return mu, std


class Critic(nn.Module):
    def __init__(self, n_obs=11, hidden_size=128):
        super(Critic, self).__init__()
        self.fc_layer = nn.Linear(n_obs, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_layer = nn.Linear(hidden_size, 1)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        return self.linear_layer(x)


class StreamAC(nn.Module):
    def __init__(
        self,
        n_obs=11,
        n_actions=3,
        hidden_size=128,
        lr=1.0,
        gamma=0.99,
        lamda=0.8,
        kappa_policy=3.0,
        kappa_value=2.0,
    ):
        super(StreamAC, self).__init__()
        self.gamma = gamma
        self.policy_net = Actor(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size)
        self.value_net = Critic(n_obs=n_obs, hidden_size=hidden_size)
        self.optimizer_policy = Optimizer(
            self.policy_net.parameters(),
            lr=lr,
            gamma=gamma,
            lamda=lamda,
            kappa=kappa_policy,
        )
        self.optimizer_value = Optimizer(
            self.value_net.parameters(),
            lr=lr,
            gamma=gamma,
            lamda=lamda,
            kappa=kappa_value,
        )

    def pi(self, x):
        return self.policy_net(x)

    def v(self, x):
        return self.value_net(x)

    def sample_action(self, s):
        x = torch.from_numpy(s).float()
        mu, std = self.pi(x)
        dist = Normal(mu, std)
        action = dist.sample().detach().cpu().numpy()
        return  action 
        #np.clip(action, -1.0, 1.0) #dist.sample().detach().cpu().numpy()

    def update_params(self, s, a, r, s_prime, done, entropy_coeff, overshooting_info=False):
        done_mask = 0 if done else 1
        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(np.array(a), dtype=torch.float32)
        r = torch.tensor(np.array(r), dtype=torch.float32)
        s_prime = torch.tensor(np.array(s_prime), dtype=torch.float32)
        done_mask = torch.tensor(np.array(done_mask), dtype=torch.float32)

        v_s, v_prime = self.v(s), self.v(s_prime)
        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s

        mu, std = self.pi(s)
        dist = Normal(mu, std)

        log_prob_pi = -(dist.log_prob(a)).sum()
        value_output = -v_s
        entropy_pi = -entropy_coeff * dist.entropy().sum() * torch.sign(delta).item()

        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        value_output.backward()
        (log_prob_pi + entropy_pi).backward()
        self.optimizer_policy.step(delta.item(), reset=done)
        self.optimizer_value.step(delta.item(), reset=done)

        if overshooting_info:
            v_s, v_prime = self.v(s), self.v(s_prime)
            td_target = r + self.gamma * v_prime * done_mask
            delta_bar = td_target - v_s
            if torch.sign(delta_bar * delta).item() == -1:
                print("Overshooting Detected!")


def main(
    env_name,
    backend,
    seed,
    lr,
    gamma,
    lamda,
    total_steps,
    entropy_coeff,
    kappa_policy,
    kappa_value,
    debug,
    overshooting_info,
    render=False,
    results_root="results",
    algo="ac",
):
    ensure_run_dirs(results_root, backend, algo, env_name, seed)

    config = {
        "algo": algo,
        "env_name": env_name,
        "backend": backend,
        "seed": seed,
        "lr": lr,
        "gamma": gamma,
        "lamda": lamda,
        "total_steps": total_steps,
        "entropy_coeff": entropy_coeff,
        "kappa_policy": kappa_policy,
        "kappa_value": kappa_value,
        "eval_interval": 10000,
        "eval_episodes": 10,
        "hidden_size": 128,
        "uses_obgd": True,
        "uses_layer_norm": True,
        "uses_sparse_init": True,
        "uses_reward_scaling_train": True,
        "uses_obs_norm": True,
        "uses_avg": False,
        "uses_penult_norm": False,
        "uses_td_scaling": False,
    }

    save_run_config(results_root, backend, algo, env_name, seed, config)

    train_logger = CSVLogger(
        train_csv_path(results_root, backend, algo, env_name, seed),
        ["step", "episode_return"],
    )

    eval_logger = CSVLogger(
        eval_csv_path(results_root, backend, algo, env_name, seed),
        ["step", "eval_return_mean", "eval_return_std"],
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_train_env(
        env_name=env_name,
        backend=backend,
        gamma=gamma,
        render=render,
    )

    agent = StreamAC(
        n_obs=env.observation_space.shape[0],
        n_actions=env.action_space.shape[0],
        lr=lr,
        gamma=gamma,
        lamda=lamda,
        kappa_policy=kappa_policy,
        kappa_value=kappa_value,
    )

    if debug:
        env_id = getattr(getattr(env, "spec", None), "id", env_name)
        print(f"seed: {seed} env: {env_id} backend: {backend}")

    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
#================================================================================
    eval_interval = 10000
    eval_episodes = 10
#================================================================================ 
    for t in range(1, total_steps + 1):
        a = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        agent.update_params(s, a, r, s_prime, done, entropy_coeff, overshooting_info)
        s = s_prime

        if t % eval_interval == 0:
            eval_mean, eval_std = evaluate_policy(
                agent=agent,
                env_name=env_name,
                backend=backend,
                train_env=env,
                episodes=eval_episodes,
                seed=seed,
            )

            eval_logger.log({
                "step": t,
                "eval_return_mean": eval_mean,
                "eval_return_std": eval_std,
            })
            print(f"[EVAL] step = {t} mean = {eval_mean:.2f} std = {eval_std:.2f}")

        if done:
            episode_return = float(np.asarray(info["episode"]["r"]).item())

            if debug:
                print(f"Episodic Return: {episode_return}, Time Step {t}")

            returns.append(episode_return)
            term_time_steps.append(t)

            train_logger.log({
                "step": t,
                "episode_return": episode_return,
            })

            s, _ = env.reset()

    env.close()
    train_logger.close()
    eval_logger.close()

    with open(returns_pkl_path(results_root, backend, algo, env_name, seed), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream AC(λ)")
    parser.add_argument("--env_name", type=str, default="HalfCheetah-v4")
    parser.add_argument("--backend", type=str, default="mujoco", choices=["mujoco", "dmcontrol"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.8)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    parser.add_argument("--kappa_policy", type=float, default=3.0)
    parser.add_argument("--kappa_value", type=float, default=2.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overshooting_info", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--results_root", type=str, default="results")
    args = parser.parse_args()

    main(
        env_name=args.env_name,
        backend=args.backend,
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        lamda=args.lamda,
        total_steps=args.total_steps,
        entropy_coeff=args.entropy_coeff,
        kappa_policy=args.kappa_policy,
        kappa_value=args.kappa_value,
        debug=args.debug,
        overshooting_info=args.overshooting_info,
        render=args.render,
        results_root=args.results_root,
        algo="ac",
    )