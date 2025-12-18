import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_scale_dir(scale_dir: str):
    csvs = sorted(glob.glob(os.path.join(scale_dir, "eval_seed*.csv")))
    if not csvs:
        return None

    dfs = [pd.read_csv(p) for p in csvs]
    for p, d in zip(csvs, dfs):
        if "step" not in d.columns or "return" not in d.columns:
            raise ValueError(f"CSV missing required columns (step, return): {p}")

    steps = dfs[0]["step"].to_numpy()
    returns = np.stack([d["return"].to_numpy() for d in dfs], axis=0)  # (n_seeds, T)

    mean_returns = returns.mean(axis=0)
    std_returns = returns.std(axis=0, ddof=1) if returns.shape[0] > 1 else np.zeros_like(mean_returns)
    n_seeds = returns.shape[0]
    return steps, mean_returns, std_returns, n_seeds


def get_run_dir() -> str:
    latest_path = os.path.join("runs", "latest_run.txt")
    if os.path.exists(latest_path):
        with open(latest_path, "r", encoding="utf-8") as f:
            run_dir = os.path.normpath(f.read().strip())
        if os.path.isdir(run_dir):
            return run_dir

    run_dirs = [p for p in glob.glob(os.path.join("runs", "*")) if os.path.isdir(p)]
    if not run_dirs:
        raise FileNotFoundError("No run directories found under runs/. Run training first.")
    return os.path.normpath(sorted(run_dirs)[-1])


def main():
    run_dir = get_run_dir()
    scale_dirs = sorted(glob.glob(os.path.join(run_dir, "scale_*")))
    if not scale_dirs:
        raise FileNotFoundError(f"No scale_* directories found in run_dir: {run_dir}")

    plt.figure()
    plotted = 0

    for scale_dir in scale_dirs:
        scale = scale_dir.split("_")[-1]
        data = load_scale_dir(scale_dir)
        if data is None:
            continue

        steps, mean_ret, std_ret, n_seeds = data

        # 95% confidence interval for the mean
        ci = 1.96 * std_ret / np.sqrt(n_seeds) if n_seeds > 1 else np.zeros_like(mean_ret)
        lo = mean_ret - ci
        hi = mean_ret + ci

        plt.plot(steps, mean_ret, label=f"scale={scale}")
        plt.fill_between(steps, lo, hi, alpha=0.2)
        plotted += 1

    if plotted == 0:
        raise FileNotFoundError(f"No eval_seed*.csv files found under: {run_dir}")

    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return (deterministic)")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(run_dir, "reward_scale_sweep.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
