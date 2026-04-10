
import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_RAW = "results/raw"
RESULTS_FIG = "results/figures"
RESULTS_AGG = "results/aggregated"

os.makedirs(RESULTS_FIG, exist_ok=True)
os.makedirs(RESULTS_AGG, exist_ok=True)


def moving_average(values, window=3):
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return values
    out = np.convolve(values, np.ones(window) / window, mode="valid")
    pad = np.full(window - 1, out[0])
    return np.concatenate([pad, out])


def aggregate_eval(env_name):
    files = sorted(glob.glob(f"{RESULTS_RAW}/{env_name}_eval_seed*.csv"))
    if not files:
        raise FileNotFoundError(f"Keine Eval-Dateien gefunden für {env_name}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        seed = os.path.basename(f).split("seed")[-1].split(".")[0]
        df["seed"] = int(seed)
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)

    grouped = full.groupby("step")["eval_return_mean"]
    mean = grouped.mean()
    std = grouped.std(ddof=1)
    n = grouped.count()
    ci95 = 1.96 * (std / np.sqrt(n))

    agg = pd.DataFrame({
        "step": mean.index,
        "mean": mean.values,
        "std": std.values,
        "n": n.values,
        "ci95_low": (mean - ci95).values,
        "ci95_high": (mean + ci95).values,
    })

    agg.to_csv(f"{RESULTS_AGG}/{env_name}_eval_aggregated.csv", index=False)
    return full, agg


def aggregate_train(env_name):
    files = sorted(glob.glob(f"{RESULTS_RAW}/{env_name}_train_seed*.csv"))
    if not files:
        raise FileNotFoundError(f"Keine Train-Dateien gefunden für {env_name}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        seed = os.path.basename(f).split("seed")[-1].split(".")[0]
        df["seed"] = int(seed)
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)

    grouped = full.groupby("step")["episode_return"]
    mean = grouped.mean()
    std = grouped.std(ddof=1)
    n = grouped.count()
    ci95 = 1.96 * (std / np.sqrt(n))

    agg = pd.DataFrame({
        "step": mean.index,
        "mean": mean.values,
        "std": std.values,
        "n": n.values,
        "ci95_low": (mean - ci95).values,
        "ci95_high": (mean + ci95).values,
    })

    agg.to_csv(f"{RESULTS_AGG}/{env_name}_train_aggregated.csv", index=False)
    return full, agg


def plot_eval(env_name, agg, smooth_window=1):
    x = agg["step"].to_numpy()
    mean = agg["mean"].to_numpy()
    low = agg["ci95_low"].to_numpy()
    high = agg["ci95_high"].to_numpy()

    if smooth_window > 1:
        mean = moving_average(mean, smooth_window)
        low = moving_average(low, smooth_window)
        high = moving_average(high, smooth_window)

    plt.figure(figsize=(8, 5))
    plt.plot(x, mean, label="Mean evaluation return")
    plt.fill_between(x, low, high, alpha=0.3, label="95% CI")
    plt.xlabel("Environment Steps")
    plt.ylabel("Evaluation Return")
    plt.title(f"Evaluation: Stream AC(lambda) on {env_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_FIG}/{env_name}_eval_plot.png", dpi=200)
    plt.close()


def plot_train(env_name, agg, smooth_window=3):
    x = agg["step"].to_numpy()
    mean = agg["mean"].to_numpy()
    low = agg["ci95_low"].to_numpy()
    high = agg["ci95_high"].to_numpy()

    if smooth_window > 1:
        mean = moving_average(mean, smooth_window)
        low = moving_average(low, smooth_window)
        high = moving_average(high, smooth_window)

    plt.figure(figsize=(8, 5))
    plt.plot(x, mean, label="Mean training return")
    plt.fill_between(x, low, high, alpha=0.3, label="95% CI")
    plt.xlabel("Environment Steps")
    plt.ylabel("Training Episode Return")
    plt.title(f"Training: Stream AC(lambda) on {env_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_FIG}/{env_name}_train_plot.png", dpi=200)
    plt.close()


def save_summary(env_name, eval_agg):
    final_row = eval_agg.iloc[-1]
    summary = pd.DataFrame([{
        "environment": env_name,
        "final_step": int(final_row["step"]),
        "final_mean_eval_return": float(final_row["mean"]),
        "final_std_eval_return": float(final_row["std"]),
        "final_ci95_low": float(final_row["ci95_low"]),
        "final_ci95_high": float(final_row["ci95_high"]),
    }])
    summary.to_csv(f"{RESULTS_AGG}/{env_name}_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    args = parser.parse_args()

    _, eval_agg = aggregate_eval(args.env_name)
    _, train_agg = aggregate_train(args.env_name)

    plot_eval(args.env_name, eval_agg, smooth_window=1)
    plot_train(args.env_name, train_agg, smooth_window=3)
    save_summary(args.env_name, eval_agg)

    print(f"Plots saved in {RESULTS_FIG}/")
    print(f"Aggregated data saved in {RESULTS_AGG}/")


if __name__ == "__main__":
    main()
