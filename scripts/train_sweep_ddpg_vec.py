"""
Train DDPG with reward scale sweep using vectorized environments (faster).
"""
import os
import time
import yaml
import numpy as np
import gymnasium as gym
from tqdm import tqdm

from sac.utils import set_seed
from sac.buffer import ReplayBuffer
from ddpg.agent import DDPGAgent, DDPGConfig


def evaluate(env, agent: DDPGAgent, episodes: int, seed: int) -> float:
    """Evaluate agent deterministically."""
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 + ep)
        done = False
        trunc = False
        ep_ret = 0.0
        while not (done or trunc):
            act = agent.act(obs)
            obs, rew, done, trunc, _ = env.step(act)
            ep_ret += float(rew)
        returns.append(ep_ret)
    return float(np.mean(returns))


def train_one_vec(
    env_id: str, 
    cfg_dict: dict, 
    reward_scale: float, 
    seed: int, 
    out_dir: str,
    num_envs: int
):
    set_seed(seed)
    
    # Create vectorized training environments
    def make_env():
        return gym.make(env_id)
    
    envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
    eval_env = gym.make(env_id)
    
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    act_low = envs.single_action_space.low
    act_high = envs.single_action_space.high
    
    # Create agent
    agent_cfg = DDPGConfig(
        gamma=float(cfg_dict["gamma"]),
        tau=float(cfg_dict["tau"]),
        actor_lr=float(cfg_dict["lr"]),
        critic_lr=float(cfg_dict["lr"]),
        hidden=int(cfg_dict["hidden"]),
        batch_size=int(cfg_dict["batch_size"]),
        reward_scale=float(reward_scale),
    )
    agent = DDPGAgent(obs_dim, act_dim, agent_cfg)
    
    # Create replay buffer
    buffer = ReplayBuffer(obs_dim, act_dim, int(cfg_dict["replay_size"]))
    
    # Training parameters
    total_steps = int(cfg_dict["total_steps"])
    start_steps = int(cfg_dict["start_steps"])
    eval_interval = int(cfg_dict["eval_interval"])
    eval_episodes = int(cfg_dict["eval_episodes"])
    updates_per_iter = int(cfg_dict.get("updates_per_iter", 1))
    exploration_noise = float(cfg_dict.get("exploration_noise", 0.1))
    
    # Setup logging
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"eval_seed{seed}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("step,return\n")
    
    # Training loop
    obs, _ = envs.reset(seed=seed)
    step_count = 0
    last_eval_return = 0.0
    
    # Progress bar
    pbar = tqdm(
        total=total_steps,
        desc=f"scale={reward_scale:.1f} seed={seed}",
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] '
    )
    
    while step_count < total_steps:
        # Select actions for all environments
        if step_count <= start_steps:
            acts = np.array([envs.single_action_space.sample() for _ in range(num_envs)])
        else:
            acts = np.array([agent.act(obs[i]) for i in range(num_envs)])
            acts += np.random.normal(0.0, exploration_noise, size=acts.shape)
            acts = np.clip(acts, act_low, act_high)

        # Step all environments
        next_obs, rews, dones, truncs, _ = envs.step(acts)
        
        # Store transitions
        for i in range(num_envs):
            d = float(dones[i] or truncs[i])
            buffer.add(obs[i], acts[i], rews[i], next_obs[i], d)
        
        obs = next_obs
        step_count += num_envs
        
        # Update agent multiple times per iteration
        if buffer.size >= agent_cfg.batch_size:
            for _ in range(updates_per_iter):
                batch = buffer.sample(agent_cfg.batch_size)
                agent.update(batch)
        
        # Evaluate periodically
        if step_count % eval_interval < num_envs:
            ret = evaluate(eval_env, agent, eval_episodes, seed)
            last_eval_return = ret
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{step_count},{ret}\n")
            pbar.set_postfix({
                'eval_return': f'{ret:.1f}',
                'buffer': f'{buffer.size}/{buffer.capacity}'
            })
        
        # Update progress bar
        pbar.update(num_envs)
    
    pbar.close()
    
    envs.close()
    eval_env.close()


def main():
    # Change this line to select environment
    with open("configs/figure2-humanoid.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    env_id = cfg["env_id"]
    seeds = cfg["seeds"]
    reward_scale = cfg["reward_scale"]
    num_envs = int(cfg.get("num_envs", 4))
    
    run_dir = os.path.join("runs", f"{env_id}_ddpg_vec_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)
    
    print("="*70)
    print(f"DDPG Reward Scale Sweep - {env_id}")
    print("="*70)
    print(f"Parallel environments: {num_envs}")
    print(f"Seeds: {seeds}")
    print(f"Reward scale: {reward_scale}")
    print(f"Total experiments: {len(seeds)}")
    print(f"Output: {run_dir}")
    print("="*70)
    print()
    
    experiment_num = 0
    total_experiments = len(seeds)
    
    for scale in [reward_scale]:
        scale_dir = os.path.join(run_dir, f"scale_{scale}")
        for seed in seeds:
            experiment_num += 1
            print(f"\n[Experiment {experiment_num}/{total_experiments}]")
            train_one_vec(env_id, cfg, reward_scale=float(scale), seed=int(seed), 
                         out_dir=scale_dir, num_envs=num_envs)
    
    # Save reference to latest run
    with open("runs/latest_run.txt", "w", encoding="utf-8") as f:
        f.write(run_dir)
    
    print("\n" + "="*70)
    print(f"[OK] All experiments complete!")
    print(f"Results saved to: {run_dir}")
    print("="*70)
    print("\nNext steps:")
    print("  python scripts/plot_sweep.py       # Plot single environment")
    print("  python scripts/plot_multi_sweep.py # Compare multiple environments")


if __name__ == "__main__":
    main()

