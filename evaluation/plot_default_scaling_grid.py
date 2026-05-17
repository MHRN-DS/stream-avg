import argparse
import os
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import sanitize_name


@dataclass(frozen=True)
class EnvSpec:
    backend: str
    env_name: str
    action_suffix: str = ""


ENVS = [
    EnvSpec("mujoco", "HalfCheetah-v4"),
    EnvSpec("mujoco", "Hopper-v4"),
    EnvSpec("mujoco", "Walker2d-v4"),
    EnvSpec("mujoco", "Ant-v4"),
    EnvSpec("dmcontrol", "cheetah-run"),
    EnvSpec("dmcontrol", "walker-walk"),
    EnvSpec("dmcontrol", "dog-walk"),
    EnvSpec("dmcontrol", "finger-spin-v0", "clip_"),
]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return values
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    pad = np.full(window - 1, smoothed[0])
    return np.concatenate([pad, smoothed])


def aggregate_eval(results_root: str, spec: EnvSpec, algo: str, seeds: Iterable[int]) -> pd.DataFrame:
    frames = []
    safe_env_name = sanitize_name(spec.env_name)
    for seed in seeds:
        path = os.path.join(results_root, spec.backend, algo, safe_env_name, f"seed_{seed}", "eval.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frames.append(frame)

    full = pd.concat(frames, ignore_index=True)
    grouped = full.groupby("step")["eval_return_mean"]
    mean = grouped.mean()
    std = grouped.std(ddof=1).fillna(0.0)
    n = grouped.count()
    ci95 = 1.96 * (std / np.sqrt(n)).fillna(0.0)
    return pd.DataFrame(
        {
            "step": mean.index.to_numpy(),
            "mean": mean.to_numpy(),
            "ci95_low": (mean - ci95).to_numpy(),
            "ci95_high": (mean + ci95).to_numpy(),
        }
    )


def plot_panel(
    ax,
    results_root: str,
    spec: EnvSpec,
    family: str,
    use_layer_norm: bool,
    seeds: Iterable[int],
    smooth: int,
) -> None:
    if use_layer_norm:
        variants = [
            (f"avg_fix_{family}_{spec.action_suffix}ln", "AVG + LayerNorm"),
            (f"avg_fix_{family}_{spec.action_suffix}reward_ln", "Reward Scaling + LayerNorm"),
            (f"avg_fix_{family}_{spec.action_suffix}reward_eta_ln", "Reward + Eta Scaling + LayerNorm"),
        ]
        title = f"{spec.env_name} Comparison of Different Scaling (LayerNorm)"
    else:
        variants = [
            (f"avg_fix_{family}_{spec.action_suffix}vanilla", "AVG"),
            (f"avg_fix_{family}_{spec.action_suffix}reward", "Reward Scaling"),
            (f"avg_fix_{family}_{spec.action_suffix}reward_eta", "Reward + Eta Scaling"),
        ]
        title = f"{spec.env_name} Comparison of Different Scaling"

    for algo, label in variants:
        agg = aggregate_eval(results_root, spec, algo, seeds)
        x = agg["step"].to_numpy()
        mean = moving_average(agg["mean"].to_numpy(), smooth)
        low = moving_average(agg["ci95_low"].to_numpy(), smooth)
        high = moving_average(agg["ci95_high"].to_numpy(), smooth)
        line, = ax.plot(x, mean, linewidth=1.8, label=label)
        ax.fill_between(x, low, high, alpha=0.16, color=line.get_color(), linewidth=0)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Environment Steps", fontsize=9)
    ax.set_ylabel("Evaluation Return", fontsize=9)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(fontsize=7, frameon=False, loc="best")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 4x4 AVG scaling comparison grid.")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--family", type=str, default="default", choices=["default", "specific"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--pdf_output", type=str, default=None)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/comparisons/avg_{args.family}_scaling_4x4_eval_grid.png"
    if args.pdf_output is None:
        args.pdf_output = f"results/comparisons/avg_{args.family}_scaling_4x4_eval_grid.pdf"

    fig, axes = plt.subplots(4, 4, figsize=(26, 22), constrained_layout=True)
    for pair_index, spec in enumerate(ENVS):
        row = pair_index // 2
        col = (pair_index % 2) * 2
        plot_panel(axes[row, col], args.results_root, spec, args.family, False, args.seeds, args.smooth)
        plot_panel(axes[row, col + 1], args.results_root, spec, args.family, True, args.seeds, args.smooth)

    family_title = "Default Adam Hyperparameters" if args.family == "default" else "Specific Hyperparameters"
    fig.suptitle(f"AVG {family_title}: Scaling and LayerNorm Comparison", fontsize=18)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=240)
    if args.pdf_output:
        os.makedirs(os.path.dirname(args.pdf_output), exist_ok=True)
        fig.savefig(args.pdf_output)
    plt.close(fig)
    print(f"Saved grid plot to: {args.output}")
    if args.pdf_output:
        print(f"Saved PDF grid plot to: {args.pdf_output}")


if __name__ == "__main__":
    main()
