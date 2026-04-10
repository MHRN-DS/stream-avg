# implementation of the Stream AC(λ) algorithm with AVG-style TD error (RMS-based) scaling, adapted for continuous control tasks.

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

"""
This module implements a streaming actor-critic algorithm for continuous action spaces, with optional use of average

To run the algorithm, pass this in terminal:

 python -m algorithms.stream_ac_continuous_avg \
  --env_name Hopper-v4 \
  --seed 1 \
  --use_avg \
  --use_penult_norm

"""

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

def save_run_config(results_root, backend, algo, env_name, seed, config):
    run_path = ensure_run_dirs(results_root, backend, algo, env_name, seed)
    config_path = Path(run_path) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent = 2, sort_keys = True)

class RunningTDScale:
    def __init__(self, beta=0.999, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.sq_mean = 1.0

    def update(self, x: float):
        x2 = float(x) * float(x)
        self.sq_mean = self.beta * self.sq_mean + (1.0 - self.beta) * x2

    @property
    def scale(self) -> float:
        return (self.sq_mean + self.eps) ** 0.5

class Actor(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, use_penult_norm=False):
        super(Actor, self).__init__()
        self.use_penult_norm = use_penult_norm
        self.fc_layer = nn.Linear(n_obs, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_mu = nn.Linear(hidden_size, n_actions)
        self.linear_std = nn.Linear(hidden_size, n_actions)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x)
        if not self.use_penult_norm:
            x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)

        x = self.hidden_layer(x)
        if not self.use_penult_norm:
            x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)

        if self.use_penult_norm:
            x = x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-8)

        mu = self.linear_mu(x)
        pre_std = self.linear_std(x)
        std = F.softplus(pre_std)
        return mu, std

class Critic(nn.Module):
    def __init__(self, n_obs=11, hidden_size=128, use_penult_norm=False):
        super(Critic, self).__init__()
        self.use_penult_norm = use_penult_norm
        self.fc_layer = nn.Linear(n_obs, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_layer = nn.Linear(hidden_size, 1)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x)
        if not self.use_penult_norm:
            x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)

        x = self.hidden_layer(x)
        if not self.use_penult_norm:
            x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)

        if self.use_penult_norm:
            x = x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-8)

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
        use_avg=False,
        use_penult_norm=False,
        td_scale_beta=0.999,
        td_scale_eps=1e-8,
    ):
        super(StreamAC, self).__init__()
        self.gamma = gamma
        self.use_avg = use_avg

        self.policy_net = Actor(
            n_obs=n_obs,
            n_actions=n_actions,
            hidden_size=hidden_size,
            use_penult_norm=use_penult_norm,
        )
        self.value_net = Critic(
            n_obs=n_obs,
            hidden_size=hidden_size,
            use_penult_norm=use_penult_norm,
        )

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

        self.td_scale = RunningTDScale(beta=td_scale_beta, eps=td_scale_eps)

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

        v_s = self.v(s)
        v_prime = self.v(s_prime)
        td_target = r + self.gamma * v_prime * done_mask
        delta_raw = td_target - v_s

        mu, std = self.pi(s)
        dist = Normal(mu, std)

        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()

        if not self.use_avg:
            # EXACT baseline path
            log_prob_pi = -(dist.log_prob(a)).sum()
            value_output = -v_s
            entropy_pi = -entropy_coeff * dist.entropy().sum() * torch.sign(delta_raw).item()

            value_output.backward()
            (log_prob_pi + entropy_pi).backward()

            self.optimizer_policy.step(delta_raw.item(), reset=done)
            self.optimizer_value.step(delta_raw.item(), reset=done)

            td_scale = 1.0
            td_used = delta_raw

        else:

            # AVG path
            self.td_scale.update(delta_raw.item())

            #td_scale = self.td_scale.scale + 1e-6 # add small constant to prevent extreme scaling when td errors are very small in the beginning of training, but amplifies the TD error when td errors are large and small, which can help stabilize training in the early phase when value estimates are very inaccurate. Note that this means there is effectively no scaling (i.e. scaling factor of 1.0) when td errors are small, but allows for scaling when td errors are large.
            #td_scale = max(td_scale, 0.1) # just ampliies gain, so we add epsilon

            #td_scale = max(self.td_scale.scale, 1e-6) # prevent extreme scaling when td errors are very small in the beginning of training, but amplifies the TD error when td errors are large and small, which can help stabilize training in the early phase when value estimates are very inaccurate. Note that this means there is effectively no scaling (i.e. scaling factor of 1.0) when td errors are small, but allows for scaling when td errors are large.
            td_scale = max(self.td_scale.scale, 1e-6) # allow scaling below 1.0 for small TD errors

            td_used = delta_raw / td_scale
            #td_used = torch.clamp(td_used, -10.0, 10.0) # clip to prevent extreme updates which can destabilize training
            log_prob_pi = -(dist.log_prob(a)).sum()
            entropy_pi = -entropy_coeff * dist.entropy().sum() * torch.sign(delta_raw).item()

            value_output = -v_s
            policy_output = log_prob_pi + entropy_pi

            value_output.backward()
            policy_output.backward()

            step_signal = td_used.item()
            self.optimizer_policy.step(step_signal, reset=done)
            self.optimizer_value.step(step_signal, reset=done)

        if overshooting_info:
            v_s_new, v_prime_new = self.v(s), self.v(s_prime)
            td_target_new = r + self.gamma * v_prime_new * done_mask
            delta_bar = td_target_new - v_s_new
            if torch.sign(delta_bar * delta_raw).item() == -1:
                print("Overshooting Detected!")

        return {
            "td_raw": float(delta_raw.item()),
            "td_used": float(td_used), # epsilon to prevent extreme values, and also clip to prevent extreme updates which can destabilize training
            "td_scale": float(td_scale),
        }
    

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
    algo="ac_avg",
    use_avg=False,
    use_penult_norm=False,
    hidden_size=128,
    eval_interval=10000,
    eval_episodes=10,
    td_scale_beta=0.999,
    td_scale_eps=1e-8,
):
    run_path = ensure_run_dirs(results_root, backend, algo, env_name, seed)

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
        "eval_interval": eval_interval,
        "eval_episodes": eval_episodes,
        "hidden_size": hidden_size,
        "uses_obgd": True,
        "uses_layer_norm": True,
        "uses_sparse_init": True,
        "uses_reward_scaling_train": True,
        "uses_obs_norm": True,
        "uses_avg": use_avg,
        "uses_penult_norm": use_penult_norm,
        "uses_td_scaling": use_avg,
        "td_scale_beta": td_scale_beta,
        "td_scale_eps": td_scale_eps,
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

    diag_logger = CSVLogger(
        str(Path(run_path) / "diagnostics.csv"),
        ["step", "td_raw", "td_used", "td_scale"],
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
        hidden_size=hidden_size,
        lr=lr,
        gamma=gamma,
        lamda=lamda,
        kappa_policy=kappa_policy,
        kappa_value=kappa_value,
        use_avg=use_avg,
        use_penult_norm=use_penult_norm,
        td_scale_beta=td_scale_beta,
        td_scale_eps=td_scale_eps,
    )

    if debug:
        env_id = getattr(getattr(env, "spec", None), "id", env_name)
        print(f"seed: {seed} env: {env_id} backend: {backend} algo: {algo}")

    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)

    last_diag = {"td_raw": np.nan, "td_used": np.nan, "td_scale": np.nan}

    for t in range(1, total_steps + 1):
        a = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        last_diag = agent.update_params(s, a, r, s_prime, done, entropy_coeff, overshooting_info)
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

            diag_logger.log({
                "step": t,
                "td_raw": last_diag["td_raw"],
                "td_used": last_diag["td_used"],
                "td_scale": last_diag["td_scale"],
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
    diag_logger.close()

    with open(returns_pkl_path(results_root, backend, algo, env_name, seed), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Stream AC(λ) + AVG")
    parser.add_argument("--env_name", type = str, default = "HalfCheetah-v4")
    parser.add_argument("--backend", type = str, default = "mujoco", choices = ["mujoco", "dmcontrol"])
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--lr", type = float, default = 1.0)
    parser.add_argument("--gamma", type = float, default = 0.99)
    parser.add_argument("--lamda", type = float, default = 0.8)
    parser.add_argument("--total_steps", type = int, default = 100000)
    parser.add_argument("--entropy_coeff", type = float, default = 0.01)
    parser.add_argument("--kappa_policy", type = float, default = 3.0)
    parser.add_argument("--kappa_value", type = float, default = 2.0)
    parser.add_argument("--debug", action = "store_true")
    parser.add_argument("--overshooting_info", action = "store_true")
    parser.add_argument("--render", action = "store_true")
    parser.add_argument("--results_root", type = str, default="results")
    parser.add_argument("--algo", type = str, default = "ac_avg")
    parser.add_argument("--use_penult_norm", action = "store_true")
    parser.add_argument("--use_avg", action = "store_true")
    parser.add_argument("--hidden_size", type = int, default = 128)
    parser.add_argument("--eval_interval", type = int, default = 10000)
    parser.add_argument("--eval_episodes", type = int, default = 10)
    parser.add_argument("--td_scale_beta", type = float, default = 0.999)
    parser.add_argument("--td_scale_eps", type = float, default = 1e-8)
    args = parser.parse_args()

    main(
        env_name = args.env_name,
        backend = args.backend,
        seed = args.seed,
        lr = args.lr,
        gamma = args.gamma,
        lamda = args.lamda,
        total_steps = args.total_steps,
        entropy_coeff = args.entropy_coeff,
        kappa_policy = args.kappa_policy,
        kappa_value = args.kappa_value,
        debug = args.debug,
        overshooting_info = args.overshooting_info,
        render = args.render,
        results_root = args.results_root,
        algo = args.algo,
        use_avg = args.use_avg,
        use_penult_norm = args.use_penult_norm,
        hidden_size = args.hidden_size,
        eval_interval = args.eval_interval,
        eval_episodes = args.eval_episodes,
        td_scale_beta = args.td_scale_beta,
        td_scale_eps = args.td_scale_eps,
    )

    # For calling from command line, use:
    # python -m algorithms.stream_ac_cont_avg \
    #   --env_name Hopper-v4 \
    #   --seed 1 \
    #   --use_avg \
    #   