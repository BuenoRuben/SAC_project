# SAC Reward Scale Sensitivity Study (Single README)

This repository implements **Soft Actor-Critic (SAC, 2018 version)** and reproduces the paper’s key empirical message:
**SAC is sensitive to reward scaling**, and the best reward scale is **environment-dependent**.

**Research question:** *How does `reward_scale` affect SAC performance (learning speed, final return, variance across seeds) on MuJoCo continuous-control tasks?*

---

## 1) What you built (scope)

- SAC implementation (2018 formulation with **Actor**, **Twin Q-functions**, **Value** + **Target Value** network)
- Off-policy training with a replay buffer
- Reward-scale sweep across multiple seeds
- Vectorized environment training for faster sampling on CPU
- Plotting utilities (single env + multi env)

---

## 2) Quickstart (Windows-friendly)

### 2.1 Install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2.3 Train sweep (recommended: vectorized)

```bash
python -m scripts.train_sweep_vec
```

This writes logs under `runs/<RUN_DIR>/...` and stores the last run path in:
- `runs/latest_run.txt`

### 2.4 Plot latest sweep

```bash
python -m scripts.plot_sweep
```

Outputs:
- `runs/<RUN_DIR>/reward_scale_sweep.png`

### 2.5 Compare multiple environments (after running multiple sweeps)

```bash
python -m scripts.plot_multi_sweep
```

---

## 3) How to switch environment

Your environment choice is driven by the YAML config loaded by the training script.

Configs live in `configs/` (examples):
- `configs/ant_sweep.yaml`
- `configs/hopper_sweep.yaml`
- `configs/walker2d_sweep.yaml`

To switch env:
1) open `scripts/train_sweep_vec.py` (or `scripts/train_sweep.py`)
2) change the config path it loads (e.g. `ant_sweep.yaml` → `hopper_sweep.yaml`)
3) run training + plot again

---

## 4) Core idea: why reward scaling matters

In SAC, the critic target includes the (scaled) reward:

\[
y = \text{reward\_scale} \cdot r + \gamma \, V_{\text{target}}(s')
\]

Reward scaling changes the relative strength of the reward term compared to the entropy-regularized terms.

Intuition:
- **reward_scale too low** → entropy dominates → policy stays too stochastic → weak exploitation
- **reward_scale too high** → reward dominates → reduced exploration / possible instability
- **best scale is environment-dependent** (task reward magnitudes differ)

---

## 5) Compute reality (why GPU often doesn’t help much)

MuJoCo training is typically **CPU-bound** because the physics simulator (`env.step`) dominates runtime.
GPU helps the neural net forward/backward passes, but if simulation is the bottleneck you won’t see big speedups.

Best speed knobs on a laptop:
- `num_envs` (parallel sampling via vectorized envs)
- `updates_per_iter` (how many gradient updates you do per sampling iteration)
- `eval_episodes` (reduce to 2 for speed; increase for final reporting)

---



## 6) Repository structure

```
SAC_project/
├── sac/
│   ├── agent.py
│   ├── networks.py
│   ├── buffer.py
│   └── utils.py
├── scripts/
│   ├── train_sweep.py
│   ├── train_sweep_vec.py
│   ├── plot_sweep.py
│   ├── plot_multi_sweep.py
│   └── visualize_project.py
├── configs/
└── runs/
```

---

## 7) References

- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
  **Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor**. ICML 2018.

- Haarnoja, T., et al. (2019).
  **Soft Actor-Critic Algorithms and Applications**. arXiv:1812.05905. (automatic temperature tuning)
