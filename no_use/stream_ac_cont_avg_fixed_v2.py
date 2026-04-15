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

# Testing:

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

def save_run_config(results_root, backend, algo, env_name, seed, config):
    run_path = ensure_run_dirs(results_root, backend, algo, env_name, seed)
    config_path = Path(run_path) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent = 2, sort_keys = True)


class RunningMoments:
    """Online running mean/variance using Welford updates."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float):
        x = float(x)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def var(self) -> float:
        if self.n < 2:
            return 0.0
        return self.m2 / self.n


class SimpleTDScaler:
    """Simple RMS-based TD scaler as fallback."""

    def __init__(self, beta: float = 0.999, eps: float = 1e-8):
        self.beta = beta
        self.eps = eps
        self.running_ms = 0.0
        self.step_count = 0

    def update(self, td_error: float):
        self.step_count += 1
        self.running_ms = self.beta * self.running_ms + (1 - self.beta) * (td_error ** 2)

    @property
    def scale(self) -> float:
        if self.step_count < 100:  # Wait for some statistics
            return 1.0
        return (self.running_ms + self.eps) ** 0.5


class ScaleTDErrorAVG:
    """
    AVG Algorithm 3 adaptation for state-value AC(lambda).

    Tracks:
      - reward stats
      - gamma stats
      - squared episode return stats

    Per-step call:    update_step(reward, gamma_t)
    End-of-episode:   update_episode_end(episode_return)

    Scale:
      sigma_delta = sqrt(Var[R] + E[G^2] * Var[gamma])
    """

    def __init__(self, eps: float = 1e-8, min_scale: float = 1e-6):
        self.reward_stats = RunningMoments()
        self.gamma_stats = RunningMoments()
        self.g2_stats = RunningMoments()
        self.eps = eps
        self.min_scale = min_scale

    def update_step(self, reward: float, gamma_t: float):
        self.reward_stats.update(reward)
        self.gamma_stats.update(gamma_t)

    def update_episode_end(self, episode_return: float):
        self.g2_stats.update(float(episode_return) ** 2)

    @property
    def scale(self) -> float:
        # Match AVG cold start: no real scaling before we have return stats
        if self.g2_stats.n < 2:
            return 1.0

        sigma2 = self.reward_stats.var + self.g2_stats.mean * self.gamma_stats.var
        sigma = (sigma2 + self.eps) ** 0.5
        scale = max(float(sigma), self.min_scale)
        # Prevent extreme amplification or suppression
        return min(max(scale, 0.1), 10.0)

class Actor(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, use_penult_norm=False):
        super(Actor, self).__init__()
        self.fc_layer = nn.Linear(n_obs, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_mu = nn.Linear(hidden_size, n_actions)
        self.linear_std = nn.Linear(hidden_size, n_actions)
        self.use_penult_norm = use_penult_norm
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
        self.fc_layer = nn.Linear(n_obs, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_layer = nn.Linear(hidden_size, 1)
        self.use_penult_norm = use_penult_norm
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
        td_scale_eps=1e-8,
        td_clip=10.0,
        use_penult_norm_actor=False,
        use_penult_norm_critic=False,
        use_simple_td_scale=False,
    ):
        super(StreamAC, self).__init__()
        self.gamma = gamma
        self.use_avg = use_avg
        self.use_simple_td_scale = use_simple_td_scale
        self.td_clip = td_clip
        self.use_penult_norm_actor = use_penult_norm_actor
        self.use_penult_norm_critic = use_penult_norm_critic

        self.policy_net = Actor(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size, use_penult_norm=use_penult_norm_actor)
        self.value_net = Critic(n_obs=n_obs, hidden_size=hidden_size, use_penult_norm=use_penult_norm_critic)
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

        self.td_scale = ScaleTDErrorAVG(eps=td_scale_eps, min_scale=1e-6)
        self.simple_td_scale = SimpleTDScaler(beta=0.999, eps=td_scale_eps)
        self.current_episode_return = 0.0
        self.current_gamma_coeff = 1.0

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

        # Keep the same AC(lambda) backward structure as the stream-x baseline.
        log_prob_pi = -(dist.log_prob(a)).sum()
        entropy_pi = -entropy_coeff * dist.entropy().sum() * torch.sign(delta_raw).item()
        value_output = -v_s

        value_output.backward()
        (log_prob_pi + entropy_pi).backward()

        td_scale = 1.0
        td_used = delta_raw

        td_scale = 1.0
        td_used = delta_raw

        if self.use_avg:
            if self.use_simple_td_scale:
                # Use simple RMS scaling
                self.simple_td_scale.update(delta_raw.item())
                td_scale = self.simple_td_scale.scale
            else:
                # Use full AVG scaling
                reward_value = float(r.item())
                gamma_t = self.gamma if not done else 0.0
                self.td_scale.update_step(reward_value, gamma_t)

                # Accumulate forward-discounted episode return G = sum_{t=0}^{T-1} gamma^t r_t
                self.current_episode_return += reward_value * self.current_gamma_coeff
                self.current_gamma_coeff *= self.gamma
                if done:
                    self.td_scale.update_episode_end(self.current_episode_return)
                    self.current_episode_return = 0.0
                    self.current_gamma_coeff = 1.0

                td_scale = self.td_scale.scale
            
            td_used = torch.clamp(delta_raw / td_scale, -self.td_clip, self.td_clip)
        else:
            self.current_episode_return += float(r.item())
            if done:
                self.current_episode_return = 0.0

        # Critical fix: use the scaled TD signal for BOTH actor and critic.
        step_signal = td_used.item() if self.use_avg else delta_raw.item()
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
            "td_used": float(td_used.item()),
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
    td_scale_eps=1e-8,
    td_clip=10.0,
    use_penult_norm_actor=False,
    use_penult_norm_critic=False,
    use_simple_td_scale=False,
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
        "uses_avg": use_avg,
        "uses_penult_norm_actor": use_penult_norm_actor,
        "uses_penult_norm_critic": use_penult_norm_critic,
        "uses_td_scaling": use_avg,
        "uses_simple_td_scale": use_simple_td_scale,
        "td_scale_eps": td_scale_eps,
        "td_clip": td_clip,
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
        str(Path(ensure_run_dirs(results_root, backend, algo, env_name, seed)) / "diagnostics.csv"),
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
        lr=lr,
        gamma=gamma,
        lamda=lamda,
        kappa_policy=kappa_policy,
        kappa_value=kappa_value,
        use_avg=use_avg,
        td_scale_eps=td_scale_eps,
        td_clip=td_clip,
        use_penult_norm_actor=use_penult_norm_actor,
        use_penult_norm_critic=use_penult_norm_critic,
        use_simple_td_scale=use_simple_td_scale,
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
    parser = argparse.ArgumentParser(description="Stream AC(λ)")
    parser.add_argument("--env_name", type=str, default="HalfCheetah-v4")
    parser.add_argument("--backend", type=str, default="mujoco", choices=["mujoco", "dmcontrol"])
    parser.add_argument("--seed", type=int, default=1)
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
    parser.add_argument("--use_avg", action="store_true")
    parser.add_argument("--use_penult_norm_actor", action="store_true")
    parser.add_argument("--use_penult_norm_critic", action="store_true")
    parser.add_argument("--use_simple_td_scale", action="store_true")
    parser.add_argument("--td_scale_eps", type=float, default=1e-8)
    parser.add_argument("--td_clip", type=float, default=10.0)
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
        algo="ac_avg",
        use_avg=args.use_avg,
        td_scale_eps=args.td_scale_eps,
        td_clip=args.td_clip,
        use_penult_norm_actor=args.use_penult_norm_actor,
        use_penult_norm_critic=args.use_penult_norm_critic,
        use_simple_td_scale=args.use_simple_td_scale,
    )