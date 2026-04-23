import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paths import figures_dir, aggregated_dir, sanitize_name


def moving_average(values, window=3):
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return values
    out = np.convolve(values, np.ones(window) / window, mode="valid")
    pad = np.full(window - 1, out[0])
    return np.concatenate([pad, out])


def compute_ci(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    grouped = df.groupby("step")[value_col]
    mean = grouped.mean()
    std = grouped.std(ddof=1)
    n = grouped.count()
    ci95 = 1.96 * (std / np.sqrt(n))
    ci95 = ci95.fillna(0.0)
    std = std.fillna(0.0)
    return pd.DataFrame({
        "step": mean.index,
        "mean": mean.values,
        "std": std.values,
        "n": n.values,
        "ci95_low": (mean - ci95).values,
        "ci95_high": (mean + ci95).values,
    })


def seed_glob(results_root, backend, algo, env_name, filename):
    safe_env_name = sanitize_name(env_name)
    return os.path.join(results_root, backend, algo, safe_env_name, "seed_*", filename)


def aggregate_eval(results_root, backend, algo, env_name):
    safe_env_name = sanitize_name(env_name)
    files = sorted(glob.glob(seed_glob(results_root, backend, algo, env_name, "eval.csv")))
    if not files:
        raise FileNotFoundError(f"No eval files found for {backend}/{algo}/{safe_env_name}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        seed = int(os.path.basename(os.path.dirname(f)).split("_")[-1])
        df["seed"] = seed
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)
    agg = compute_ci(full, "eval_return_mean")

    agg_dir = aggregated_dir(results_root, backend, algo)
    agg.to_csv(os.path.join(agg_dir, f"{safe_env_name}_eval_aggregated.csv"), index=False)
    return full, agg


def aggregate_train(results_root, backend, algo, env_name, bin_size=10000):
    safe_env_name = sanitize_name(env_name)
    files = sorted(glob.glob(seed_glob(results_root, backend, algo, env_name, "train.csv")))
    if not files:
        raise FileNotFoundError(f"No train files found for {backend}/{algo}/{safe_env_name}")

    per_seed = []
    max_step = 0
    for f in files:
        df = pd.read_csv(f).sort_values("step")
        seed = int(os.path.basename(os.path.dirname(f)).split("_")[-1])
        df["seed"] = seed
        per_seed.append(df)
        max_step = max(max_step, int(df["step"].max()))

    grid = np.arange(bin_size, max_step + bin_size, bin_size)
    aligned = []
    for df in per_seed:
        seed = int(df["seed"].iloc[0])
        values = []
        for step in grid:
            upto = df[df["step"] <= step]
            if len(upto) == 0:
                values.append(np.nan)
            else:
                values.append(float(upto["episode_return"].iloc[-1]))
        aligned.append(pd.DataFrame({
            "step": grid,
            "episode_return": values,
            "seed": seed,
        }))

    full = pd.concat(aligned, ignore_index=True)
    full = full.dropna(subset=["episode_return"])
    agg = compute_ci(full, "episode_return")

    agg_dir = aggregated_dir(results_root, backend, algo)
    agg.to_csv(os.path.join(agg_dir, f"{safe_env_name}_train_aggregated.csv"), index=False)
    return full, agg


def plot_curve(agg, ylabel, title, outfile, smooth_window=5): # set to 1 for no smoothing, set to 3 for mild smoothing, set to 5 for stronger smoothing
    x = agg["step"].to_numpy()
    mean = agg["mean"].to_numpy()
    low = agg["ci95_low"].to_numpy()
    high = agg["ci95_high"].to_numpy()

    if smooth_window > 1:
        mean = moving_average(mean, smooth_window)
        low = moving_average(low, smooth_window)
        high = moving_average(high, smooth_window)

    plt.figure(figsize=(8, 5))
    plt.plot(x, mean, label="Mean")
    plt.fill_between(x, low, high, alpha=0.3, label="95% CI")
    plt.xlabel("Environment Steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def save_summary(results_root, backend, algo, env_name, eval_agg):
    safe_env_name = sanitize_name(env_name)
    final_row = eval_agg.iloc[-1]
    summary = pd.DataFrame([{
        "environment": env_name,
        "final_step": int(final_row["step"]),
        "final_mean_eval_return": float(final_row["mean"]),
        "final_std_eval_return": float(final_row["std"]),
        "final_ci95_low": float(final_row["ci95_low"]),
        "final_ci95_high": float(final_row["ci95_high"]),
    }])

    agg_dir = aggregated_dir(results_root, backend, algo)
    summary.to_csv(os.path.join(agg_dir, f"{safe_env_name}_summary.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--backend", type=str, default="mujoco", choices=["mujoco", "dmcontrol"])
    parser.add_argument("--algo", type=str, default="ac")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--train_bin_size", type=int, default=10000)
    args = parser.parse_args()

    safe_env_name = sanitize_name(args.env_name)

    _, eval_agg = aggregate_eval(args.results_root, args.backend, args.algo, args.env_name)
    _, train_agg = aggregate_train(
        args.results_root,
        args.backend,
        args.algo,
        args.env_name,
        bin_size=args.train_bin_size,
    )

    fig_dir = figures_dir(args.results_root, args.backend, args.algo)

    plot_curve(
        eval_agg,
        ylabel="Evaluation Return",
        title=f"Evaluation: AVG with TD-Error Scaling + LN + RS + Eta Scaling on {args.env_name}",
        outfile=os.path.join(fig_dir, f"{safe_env_name}_eval_plot.png"),
        smooth_window=1,
    )
    plot_curve(
        train_agg,
        ylabel="Training Episode Return",
        title=f"Training: AVG with TD-Error Scaling + LN + RS + Eta Scaling on {args.env_name}",
        outfile=os.path.join(fig_dir, f"{safe_env_name}_train_plot.png"),
        smooth_window=3,
    )
    save_summary(args.results_root, args.backend, args.algo, args.env_name, eval_agg)

    print(f"Plots saved in {fig_dir}")


if __name__ == "__main__":
    main()