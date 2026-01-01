import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_run(run_dir: str):
    scale_dirs = sorted(glob.glob(os.path.join(run_dir, "scale_*")))
    if len(scale_dirs) != 1:
        raise ValueError(f"Expected exactly one scale_* directory in {run_dir}")

    csvs = sorted(glob.glob(os.path.join(scale_dirs[0], "eval_seed*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No eval_seed*.csv files in {scale_dirs[0]}")

    dfs = [pd.read_csv(p) for p in csvs]
    steps = dfs[0]["step"].to_numpy()
    returns = np.stack([d["return"].to_numpy() for d in dfs], axis=0)

    mean = returns.mean(axis=0)
    std = returns.std(axis=0, ddof=1) if returns.shape[0] > 1 else np.zeros_like(mean)
    n = returns.shape[0]

    return steps, mean, std, n


def plot_algorithm(run_dir: str, label: str):
    steps, mean, std, n = load_run(run_dir)

    ci = 1.96 * std / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    lo = mean - ci
    hi = mean + ci

    plt.plot(steps, mean, label=label)
    plt.fill_between(steps, lo, hi, alpha=0.2)


def main():
    # ---- CHANGE THESE TWO PATHS ----
    sac_run_dir = "runs/Humanoid-v4_vec_XXXXXXXX"
    ddpg_run_dir = "runs/Humanoid-v4_ddpg_vec_YYYYYYYY"

    plt.figure(figsize=(7, 5))

    plot_algorithm(sac_run_dir, label="SAC")
    plot_algorithm(ddpg_run_dir, label="DDPG")

    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return (deterministic)")
    plt.legend()
    plt.tight_layout()

    out_path = "runs/sac_vs_ddpg.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()
