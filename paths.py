from pathlib import Path

def sanitize_name(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")

# This module provides utility functions for constructing file paths for experiment results.
def run_dir(results_root: str, backend: str, algo: str, env_name: str, seed: int) -> Path:
    safe_env_name = sanitize_name(env_name)
    return Path(results_root) / backend / algo / safe_env_name / f"seed_{seed}"

# The following functions construct specific file paths for training logs, evaluation logs, and artifacts based on the experiment configuration.
def ensure_run_dirs(results_root: str, backend: str, algo: str, env_name: str, seed: int):
    root = run_dir(results_root, backend, algo, env_name, seed)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    return root

# The following functions return the paths for training CSV, evaluation CSV, and returns pickle file for a given experiment configuration.
def train_csv_path(results_root: str, backend: str, algo: str, env_name: str, seed: int) -> str:
    return str(run_dir(results_root, backend, algo, env_name, seed) / "train.csv")

def eval_csv_path(results_root: str, backend: str, algo: str, env_name: str, seed: int) -> str:
    return str(run_dir(results_root, backend, algo, env_name, seed) / "eval.csv")

# The following function returns the path for the returns pickle file, which is used to store episode returns for later analysis.
def returns_pkl_path(results_root: str, backend: str, algo: str, env_name: str, seed: int) -> str:
    return str(run_dir(results_root, backend, algo, env_name, seed) / "artifacts" / "returns.pkl")

# The following functions construct paths for aggregated results and figures based on the experiment configuration.
def figures_dir(results_root: str, backend: str, algo: str) -> str:
    path = Path(results_root) / backend / algo / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)

def aggregated_dir(results_root: str, backend: str, algo: str) -> str:
    path = Path(results_root) / backend / algo / "aggregated"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)