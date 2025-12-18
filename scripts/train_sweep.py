import os
import time
import yaml
import numpy as np
import gymnasium as gym
from tqdm import tqdm

from sac.utils import set_seed
from sac.buffer import ReplayBuffer
from sac.agent import SACAgent, SACConfig


def evaluate(env, agent: SACAgent, episodes: int, seed: int) -> float:
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 + ep)
        done = False
        trunc = False
        ep_ret = 0.0
        while not (done or trunc):
            act = agent.act(obs, deterministic=True)
            obs, rew, done, trunc, _ = env.step(act)
            ep_ret += float(rew)
        returns.append(ep_ret)
    return float(np.mean(returns))


def train_one(env_id: str, cfg_dict: dict, reward_scale: float, seed: int, out_dir: str):
    set_seed(seed)

    env = gym.make(env_id)
    eval_env = gym.make(env_id)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    agent_cfg = SACConfig(
        gamma=float(cfg_dict["gamma"]),
        tau=float(cfg_dict["tau"]),
        lr=float(cfg_dict["lr"]),
        hidden=int(cfg_dict["hidden"]),
        batch_size=int(cfg_dict["batch_size"]),
        reward_scale=float(reward_scale),
    )
    agent = SACAgent(obs_dim, act_dim, agent_cfg)

    buffer = ReplayBuffer(obs_dim, act_dim, int(cfg_dict["replay_size"]))

    total_steps = int(cfg_dict["total_steps"])
    start_steps = int(cfg_dict["start_steps"])
    eval_interval = int(cfg_dict["eval_interval"])
    eval_episodes = int(cfg_dict["eval_episodes"])

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"eval_seed{seed}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("step,return\n")

    obs, _ = env.reset(seed=seed)
    last_eval_return = 0.0

    # Progress bar
    pbar = tqdm(
        total=total_steps,
        desc=f"scale={reward_scale:.1f} seed={seed}",
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] '
    )

    for t in range(1, total_steps + 1):
        if t <= start_steps:
            act = env.action_space.sample()
        else:
            act = agent.act(obs, deterministic=False)

        next_obs, rew, done, trunc, _ = env.step(act)
        d = float(done or trunc)
        buffer.add(obs, act, rew, next_obs, d)

        obs = next_obs
        if done or trunc:
            obs, _ = env.reset(seed=seed + t)

        if buffer.size >= agent_cfg.batch_size:
            batch = buffer.sample(agent_cfg.batch_size)
            agent.update(batch)

        if t % eval_interval == 0:
            ret = evaluate(eval_env, agent, eval_episodes, seed)
            last_eval_return = ret
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{t},{ret}\n")
            pbar.set_postfix({
                'eval_return': f'{ret:.1f}',
                'buffer': f'{buffer.size}/{buffer.capacity}'
            })
        
        pbar.update(1)
    
    pbar.close()

    env.close()
    eval_env.close()


def main():
    # Change this line to select environment
    with open("configs/ant_sweep.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_id = cfg["env_id"]
    seeds = cfg["seeds"]
    reward_scales = cfg["reward_scales"]

    run_dir = os.path.join("runs", f"{env_id}_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)

    print("="*70)
    print(f"SAC Reward Scale Sweep - {env_id}")
    print("="*70)
    print(f"Seeds: {seeds}")
    print(f"Reward scales: {reward_scales}")
    print(f"Total experiments: {len(seeds) * len(reward_scales)}")
    print(f"Output: {run_dir}")
    print("="*70)
    print()

    experiment_num = 0
    total_experiments = len(seeds) * len(reward_scales)

    for scale in reward_scales:
        scale_dir = os.path.join(run_dir, f"scale_{scale}")
        for seed in seeds:
            experiment_num += 1
            print(f"\n[Experiment {experiment_num}/{total_experiments}]")
            train_one(env_id, cfg, reward_scale=float(scale), seed=int(seed), out_dir=scale_dir)

    print("\n" + "="*70)
    print(f"[OK] All experiments complete!")
    print(f"Results saved to: {run_dir}")
    print("="*70)
    print("\nNext steps:")
    print("  python scripts/plot_sweep.py       # Plot results")
    print("  python scripts/plot_multi_sweep.py # Compare environments")


if __name__ == "__main__":
    main()
