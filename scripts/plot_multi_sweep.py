"""
Plot reward scale sensitivity across multiple environments.
Shows that reward scaling affects SAC performance universally.

Usage: python scripts/plot_multi_sweep.py
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_scale_data(run_dir: str, scale: float):
    """Load all seed data for one reward scale."""
    scale_dir = os.path.join(run_dir, f"scale_{scale}")
    if not os.path.exists(scale_dir):
        return None
    
    csv_files = glob.glob(os.path.join(scale_dir, "*.csv"))
    if not csv_files:
        return None
    
    all_returns = []
    steps = None
    
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if steps is None:
            steps = df["step"].values if "step" in df.columns else df.iloc[:, 0].values
        returns = df["return"].values if "return" in df.columns else df.iloc[:, 1].values
        all_returns.append(returns)
    
    if not all_returns:
        return None
    
    returns = np.array(all_returns)
    return {
        "steps": steps,
        "mean": returns.mean(axis=0),
        "std": returns.std(axis=0, ddof=1) if len(returns) > 1 else np.zeros_like(steps, dtype=float),
        "n_seeds": len(returns)
    }


def find_sweep_runs():
    """Find all reward scale sweep runs."""
    runs = {}
    
    # Look for run directories with scale_* subdirectories
    run_dirs = glob.glob("runs/*")
    
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue
        
        # Check if it has scale_* subdirs (indicates sweep experiment)
        scale_dirs = glob.glob(os.path.join(run_dir, "scale_*"))
        if not scale_dirs:
            continue
        
        # Try to determine environment from directory name
        run_name = os.path.basename(run_dir)
        
        # Extract environment name
        if "Hopper" in run_name:
            env_name = "Hopper"
        elif "Walker" in run_name or "walker" in run_name.lower():
            env_name = "Walker2d"
        elif "Ant" in run_name:
            env_name = "Ant"
        elif "Cheetah" in run_name or "cheetah" in run_name.lower():
            env_name = "HalfCheetah"
        else:
            env_name = run_name
        
        if env_name not in runs:
            runs[env_name] = []
        
        runs[env_name].append(run_dir)
    
    # Use most recent run for each environment
    latest_runs = {}
    for env_name, run_list in runs.items():
        latest_runs[env_name] = sorted(run_list)[-1]
    
    return latest_runs


def plot_comparison(env_runs: dict, scales: list):
    """Create comparison plot across environments."""
    n_envs = len(env_runs)
    
    if n_envs == 0:
        print("No sweep experiments found!")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 4 * ((n_envs + 1) // 2)))
    
    colors = {2.0: 'tab:orange', 5.0: 'tab:green', 10.0: 'tab:blue'}
    
    for idx, (env_name, run_dir) in enumerate(sorted(env_runs.items()), 1):
        ax = plt.subplot(2, (n_envs + 1) // 2, idx)
        
        plotted = False
        for scale in scales:
            data = load_scale_data(run_dir, scale)
            if data is None:
                continue
            
            steps = data["steps"]
            mean = data["mean"]
            std = data["std"]
            n_seeds = data["n_seeds"]
            
            # 95% confidence interval
            ci = 1.96 * std / np.sqrt(n_seeds) if n_seeds > 1 else std
            
            color = colors.get(scale, 'gray')
            ax.plot(steps, mean, label=f"scale={scale}", color=color, linewidth=2)
            ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.2)
            plotted = True
        
        if plotted:
            ax.set_title(env_name, fontsize=14, fontweight='bold')
            ax.set_xlabel("Environment Steps")
            ax.set_ylabel("Evaluation Return")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = "runs/reward_scale_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison plot: {output_path}")


def plot_final_performance(env_runs: dict, scales: list):
    """Bar chart showing final performance by environment and scale."""
    env_names = sorted(env_runs.keys())
    n_envs = len(env_names)
    
    if n_envs == 0:
        return
    
    # Collect final performance for each env and scale
    results = {scale: [] for scale in scales}
    
    for env_name in env_names:
        run_dir = env_runs[env_name]
        
        for scale in scales:
            data = load_scale_data(run_dir, scale)
            if data is not None:
                # Average over last 20% of training
                last_20_pct = int(0.8 * len(data["mean"]))
                final_perf = data["mean"][last_20_pct:].mean()
                results[scale].append(final_perf)
            else:
                results[scale].append(0)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n_envs)
    width = 0.25
    colors = {2.0: 'tab:orange', 5.0: 'tab:green', 10.0: 'tab:blue'}
    
    for i, scale in enumerate(scales):
        offset = (i - len(scales) / 2 + 0.5) * width
        ax.bar(x + offset, results[scale], width, 
               label=f'Scale {scale}', color=colors.get(scale, 'gray'))
    
    ax.set_xlabel('Environment', fontsize=12)
    ax.set_ylabel('Final Performance (last 20% avg)', fontsize=12)
    ax.set_title('Reward Scale Sensitivity Across Environments', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(env_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = "runs/reward_scale_bars.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved bar chart: {output_path}")


def create_summary_table(env_runs: dict, scales: list):
    """Create summary table of results."""
    data = []
    
    for env_name in sorted(env_runs.keys()):
        run_dir = env_runs[env_name]
        row = {"Environment": env_name}
        
        for scale in scales:
            scale_data = load_scale_data(run_dir, scale)
            if scale_data is not None:
                last_20_pct = int(0.8 * len(scale_data["mean"]))
                final_mean = scale_data["mean"][last_20_pct:].mean()
                final_std = scale_data["std"][last_20_pct:].mean()
                row[f"Scale {scale}"] = f"{final_mean:.1f} ± {final_std:.1f}"
            else:
                row[f"Scale {scale}"] = "N/A"
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_path = "runs/reward_scale_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved summary: {csv_path}")
    
    # Print to console
    print("\nSummary Table:")
    print("="*70)
    print(df.to_string(index=False))
    print()


def main():
    print("="*70)
    print("Multi-Environment Reward Scale Analysis")
    print("="*70)
    print()
    
    # Find all sweep runs
    env_runs = find_sweep_runs()
    
    if not env_runs:
        print("No reward scale sweep experiments found!")
        print("Run train_sweep.py on different environments first.")
        return
    
    print(f"Found sweep experiments for {len(env_runs)} environment(s):")
    for env_name, run_dir in sorted(env_runs.items()):
        print(f"  - {env_name}: {run_dir}")
    print()
    
    scales = [2.0, 5.0, 10.0]
    
    # Create visualizations
    print("Generating plots...")
    plot_comparison(env_runs, scales)
    plot_final_performance(env_runs, scales)
    create_summary_table(env_runs, scales)
    
    print("\n" + "="*70)
    print("✓ Analysis complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - runs/reward_scale_comparison.png   (learning curves)")
    print("  - runs/reward_scale_bars.png          (final performance)")
    print("  - runs/reward_scale_summary.csv       (numerical data)")
    print()


if __name__ == "__main__":
    main()


