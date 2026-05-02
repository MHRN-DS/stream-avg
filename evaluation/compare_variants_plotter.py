import argparse
import glob
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
How to run: 
python compare_variants_plotter.py \
  --env_name <ENV_NAME> \
  --backend <mujoco_or_dmcontrol> \
  --variants \
    <baseline folder name >= "AVG (TD-scaling)" \
    <folder name >= "A + LayerNorm" \
    <other folder name >= "B + LayerNorm" \
    <other folder name >= "C + LayerNorm" \
  --results_root results \
  --output_tag avg_consolidated
"""

try:
    from paths import figures_dir, aggregated_dir, sanitize_name  # type: ignore
except Exception:
    def sanitize_name(name: str) -> str:
        safe = name.replace('/', '_').replace('\\', '_').replace(' ', '_')
        safe = safe.replace(':', '_')
        return safe

    def figures_dir(results_root: str, backend: str, algo: str) -> str:
        path = os.path.join(results_root, backend, algo, 'figures')
        os.makedirs(path, exist_ok=True)
        return path

    def aggregated_dir(results_root: str, backend: str, algo: str) -> str:
        path = os.path.join(results_root, backend, algo, 'aggregated')
        os.makedirs(path, exist_ok=True)
        return path


def moving_average(values: np.ndarray, window: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return values
    out = np.convolve(values, np.ones(window) / window, mode="valid")
    pad = np.full(window - 1, out[0])
    return np.concatenate([pad, out])


def compute_ci(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    grouped = df.groupby("step")[value_col]
    mean = grouped.mean()
    std = grouped.std(ddof=1).fillna(0.0)
    n = grouped.count()
    ci95 = 1.96 * (std / np.sqrt(n)).fillna(0.0)
    return pd.DataFrame(
        {
            "step": mean.index,
            "mean": mean.values,
            "std": std.values,
            "n": n.values,
            "ci95_low": (mean - ci95).values,
            "ci95_high": (mean + ci95).values,
        }
    )


def seed_glob(results_root: str, backend: str, algo: str, env_name: str, filename: str) -> str:
    safe_env_name = sanitize_name(env_name)
    return os.path.join(results_root, backend, algo, safe_env_name, "seed_*", filename)


def aggregate_eval(results_root: str, backend: str, algo: str, env_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    out_dir = aggregated_dir(results_root, backend, algo)
    agg.to_csv(os.path.join(out_dir, f"{safe_env_name}_eval_aggregated.csv"), index=False)
    return full, agg


def aggregate_train(results_root: str, backend: str, algo: str, env_name: str, bin_size: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        aligned.append(pd.DataFrame({"step": grid, "episode_return": values, "seed": seed}))

    full = pd.concat(aligned, ignore_index=True)
    full = full.dropna(subset=["episode_return"])
    agg = compute_ci(full, "episode_return")

    out_dir = aggregated_dir(results_root, backend, algo)
    agg.to_csv(os.path.join(out_dir, f"{safe_env_name}_train_aggregated.csv"), index=False)
    return full, agg


def parse_variant_specs(variant_specs: List[str]) -> List[Tuple[str, str]]:
    parsed = []
    for spec in variant_specs:
        if "=" in spec:
            algo, label = spec.split("=", 1)
        else:
            algo, label = spec, spec
        parsed.append((algo.strip(), label.strip()))
    return parsed


def maybe_smooth(agg: pd.DataFrame, smooth_window: int) -> pd.DataFrame:
    out = agg.copy()
    if smooth_window > 1:
        for col in ["mean", "ci95_low", "ci95_high"]:
            out[col] = moving_average(out[col].to_numpy(), smooth_window)
    return out


def plot_multi_curve(curves: Dict[str, pd.DataFrame], ylabel: str, title: str, outfile: str, smooth_window: int = 1) -> None:
    plt.figure(figsize=(9, 5.5))
    for label, agg in curves.items():
        agg_plot = maybe_smooth(agg, smooth_window)
        x = agg_plot["step"].to_numpy()
        mean = agg_plot["mean"].to_numpy()
        low = agg_plot["ci95_low"].to_numpy()
        high = agg_plot["ci95_high"].to_numpy()
        line, = plt.plot(x, mean, label=label)
        plt.fill_between(x, low, high, alpha=0.18, color=line.get_color())

    plt.xlabel("Environment Steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=200)
    plt.close()


def build_summary(eval_aggs: Dict[str, pd.DataFrame], train_aggs: Dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    rows = []
    for label, eval_agg in eval_aggs.items():
        final_eval = eval_agg.iloc[-1]
        best_idx = eval_agg["mean"].idxmax()
        best_eval = eval_agg.loc[best_idx]
        row = {
            "variant": label,
            "final_eval_step": int(final_eval["step"]),
            "final_eval_mean": float(final_eval["mean"]),
            "final_eval_std": float(final_eval["std"]),
            "final_eval_ci95_low": float(final_eval["ci95_low"]),
            "final_eval_ci95_high": float(final_eval["ci95_high"]),
            "best_eval_step": int(best_eval["step"]),
            "best_eval_mean": float(best_eval["mean"]),
        }
        if train_aggs is not None and label in train_aggs and len(train_aggs[label]) > 0:
            final_train = train_aggs[label].iloc[-1]
            row.update(
                {
                    "final_train_step": int(final_train["step"]),
                    "final_train_mean": float(final_train["mean"]),
                    "final_train_std": float(final_train["std"]),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("final_eval_mean", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple result variants on one environment.")
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--backend", type=str, default="mujoco", choices=["mujoco", "dmcontrol"])
    parser.add_argument("--variants", type=str, nargs="+", required=True,
                        help="Variant folders to compare. Use algo or algo=Pretty Label.")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--train_bin_size", type=int, default=10000)
    parser.add_argument("--eval_smooth", type=int, default=1)
    parser.add_argument("--train_smooth", type=int, default=3)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--output_tag", type=str, default="comparison")
    args = parser.parse_args()

    safe_env_name = sanitize_name(args.env_name)
    variants = parse_variant_specs(args.variants)

    eval_aggs: Dict[str, pd.DataFrame] = {}
    train_aggs: Dict[str, pd.DataFrame] = {}
    missing_train: List[str] = []

    for algo, label in variants:
        _, eval_agg = aggregate_eval(args.results_root, args.backend, algo, args.env_name)
        eval_aggs[label] = eval_agg
        if not args.skip_train:
            try:
                _, train_agg = aggregate_train(args.results_root, args.backend, algo, args.env_name, args.train_bin_size)
                train_aggs[label] = train_agg
            except FileNotFoundError:
                missing_train.append(label)

    compare_dir = os.path.join(args.results_root, args.backend, "comparisons")
    os.makedirs(compare_dir, exist_ok=True)

    eval_out = os.path.join(compare_dir, f"{safe_env_name}_{args.output_tag}_eval.png")
    plot_multi_curve(
        eval_aggs,
        ylabel="Evaluation Return",
        title=f"Evaluation Comparison on {args.env_name}",
        outfile=eval_out,
        smooth_window=args.eval_smooth,
    )

    train_out = None
    if not args.skip_train and train_aggs:
        train_out = os.path.join(compare_dir, f"{safe_env_name}_{args.output_tag}_train.png")
        plot_multi_curve(
            train_aggs,
            ylabel="Training Episode Return",
            title=f"Training Comparison on {args.env_name}",
            outfile=train_out,
            smooth_window=args.train_smooth,
        )

    summary = build_summary(eval_aggs, train_aggs if train_aggs else None)
    summary_out = os.path.join(compare_dir, f"{safe_env_name}_{args.output_tag}_summary.csv")
    summary.to_csv(summary_out, index=False)

    print(f"Saved evaluation comparison to: {eval_out}")
    if train_out is not None:
        print(f"Saved training comparison to: {train_out}")
    if missing_train:
        print(f"Missing train.csv for: {', '.join(missing_train)}")
    print(f"Saved summary table to: {summary_out}")
    print("\nRanking by final evaluation mean:")
    print(summary[["variant", "final_eval_mean", "best_eval_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
