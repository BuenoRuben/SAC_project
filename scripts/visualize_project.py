"""
Visualize the project structure and current status.
Usage: python scripts/visualize_project.py
"""
import os
import glob


def print_tree(directory, prefix="", max_depth=3, current_depth=0, ignore_dirs=None):
    """Print directory tree structure."""
    if ignore_dirs is None:
        ignore_dirs = {"__pycache__", ".git", "venv", ".venv", "node_modules"}
    
    if current_depth >= max_depth:
        return
    
    try:
        items = sorted(os.listdir(directory))
    except PermissionError:
        return
    
    dirs = [item for item in items if os.path.isdir(os.path.join(directory, item)) 
            and item not in ignore_dirs and not item.startswith('.')]
    files = [item for item in items if os.path.isfile(os.path.join(directory, item))
             and not item.startswith('.')]
    
    # Print files first
    for i, file in enumerate(files):
        is_last = (i == len(files) - 1) and len(dirs) == 0
        connector = "+-- " if is_last else "|-- "
        print(f"{prefix}{connector}{file}")
    
    # Then directories
    for i, dir_name in enumerate(dirs):
        is_last = i == len(dirs) - 1
        connector = "+-- " if is_last else "|-- "
        print(f"{prefix}{connector}{dir_name}/")
        
        new_prefix = prefix + ("    " if is_last else "|   ")
        dir_path = os.path.join(directory, dir_name)
        print_tree(dir_path, new_prefix, max_depth, current_depth + 1, ignore_dirs)


def count_lines(file_path):
    """Count lines in a file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return len(f.readlines())
    except:
        return 0


def analyze_code():
    """Analyze code statistics."""
    python_files = glob.glob("**/*.py", recursive=True)
    python_files = [f for f in python_files if "venv" not in f and "__pycache__" not in f]
    
    total_lines = 0
    file_stats = []
    
    for file_path in python_files:
        lines = count_lines(file_path)
        total_lines += lines
        file_stats.append((file_path, lines))
    
    return total_lines, sorted(file_stats, key=lambda x: x[1], reverse=True)


def check_experiments():
    """Check what experiments have been run."""
    run_dirs = glob.glob("runs/*")
    run_dirs = [d for d in run_dirs if os.path.isdir(d)]
    
    experiments = []
    
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        
        # Count CSV files
        csv_files = glob.glob(os.path.join(run_dir, "**/*.csv"), recursive=True)
        
        # Check for plots
        png_files = glob.glob(os.path.join(run_dir, "*.png"))
        
        experiments.append({
            "name": run_name,
            "csv_count": len(csv_files),
            "has_plots": len(png_files) > 0,
            "plot_count": len(png_files)
        })
    
    return experiments


def main():
    print("="*70)
    print("SAC PROJECT STATUS")
    print("="*70)
    print()
    
    # Project structure
    print("[PROJECT STRUCTURE]")
    print("-"*70)
    print_tree(".", max_depth=2)
    print()
    
    # Code statistics
    print("="*70)
    print("[CODE STATISTICS]")
    print("-"*70)
    total_lines, file_stats = analyze_code()
    print(f"Total Python files: {len(file_stats)}")
    print(f"Total lines of code: {total_lines:,}")
    print()
    print("Top files by line count:")
    for file_path, lines in file_stats[:10]:
        print(f"  {file_path:40s} {lines:5,} lines")
    print()
    
    # Experiments
    print("="*70)
    print("[EXPERIMENTS RUN]")
    print("-"*70)
    experiments = check_experiments()
    
    if not experiments:
        print("No experiments found yet.")
        print()
        print("To run experiments:")
        print("  1. Quick test: python scripts/quick_test.py")
        print("  2. Single env: python scripts/train.py --config configs/hopper.yaml")
        print("  3. All envs:   python scripts/train_all_envs.py")
    else:
        print(f"Found {len(experiments)} experiment run(s):\n")
        for exp in experiments:
            status = "[OK] Complete" if exp["has_plots"] else "[RUNNING] In progress"
            print(f"  {status} - {exp['name']}")
            print(f"           Data files: {exp['csv_count']} CSV files")
            print(f"           Plots: {exp['plot_count']} PNG files")
            print()
    
    # Configuration files
    print("="*70)
    print("[CONFIGURATIONS]")
    print("-"*70)
    configs = glob.glob("configs/*.yaml")
    print(f"Available environment configs: {len(configs)}\n")
    for config in sorted(configs):
        config_name = os.path.basename(config).replace(".yaml", "")
        print(f"  - {config_name}")
    print()
    
    # Features implemented
    print("="*70)
    print("[FEATURES IMPLEMENTED]")
    print("-"*70)
    features = [
        ("SAC Agent", "sac/agent.py", True),
        ("Neural Networks (Actor, Q, V)", "sac/networks.py", True),
        ("Replay Buffer", "sac/buffer.py", True),
        ("Reward Scale Sweep Training", "scripts/train_sweep.py", True),
        ("Single Environment Plot", "scripts/plot_sweep.py", True),
        ("Multi-Environment Comparison", "scripts/plot_multi_sweep.py", True),
        ("Quick Test", "scripts/quick_test.py", True),
    ]
    
    for feature, file_path, implemented in features:
        status = "[OK]" if implemented and os.path.exists(file_path) else "[NO]"
        print(f"  {status} {feature:35s} ({file_path})")
    print()
    
    # Next steps
    print("="*70)
    print("[NEXT STEPS]")
    print("-"*70)
    print()
    print("1. Quick Test (5 minutes)")
    print("   python scripts/quick_test.py")
    print()
    print("2. Plot Existing Ant Results")
    print("   python scripts/plot_sweep.py")
    print()
    print("3. Run Sweep on New Environment (1.5-3 hours)")
    print("   Edit scripts/train_sweep.py line 89 to change config")
    print("   python scripts/train_sweep.py")
    print()
    print("4. Compare Multiple Environments")
    print("   python scripts/plot_multi_sweep.py")
    print()
    print("="*70)
    print("[DOCUMENTATION]")
    print("-"*70)
    docs = [
        ("README.md", "Main project documentation"),
        ("PROJECT_GUIDE.md", "Presentation guide and tips"),
        ("requirements.txt", "Python dependencies"),
    ]
    
    for doc, description in docs:
        exists = "[OK]" if os.path.exists(doc) else "[NO]"
        print(f"  {exists} {doc:25s} - {description}")
    print()
    
    print("="*70)
    print("PROJECT READY FOR EXPERIMENTS!")
    print("="*70)


if __name__ == "__main__":
    main()

