from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.optim as optim

from .networks import Actor, Critic

@dataclass
class DetSACConfig:
    gamma: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden: int = 256
    batch_size: int = 256
    reward_scale: float = 1.0
    target_update_period: int = 1000 # hard update every N env steps (or N updates)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DeterministicSACAgent:
    """
    Deterministic SAC variant (went from DDPG to this, not from SAC):
    - deterministic policy
    - two Q-functions
    - hard target updates
    - no target actor
    - no entropy term
    Exploration noise is handled outside (same as ddpg).
    """
    def __init__(self, obs_dim: int, act_dim: int, cfg: DetSACConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actor = Actor(obs_dim, act_dim, hidden=cfg.hidden).to(self.device)

        self.q1 = Critic(obs_dim, act_dim, hidden=cfg.hidden).to(self.device)
        self.q2 = Critic(obs_dim, act_dim, hidden=cfg.hidden).to(self.device)

        self.q1_targ = Critic(obs_dim, act_dim, hidden=cfg.hidden).to(self.device)
        self.q2_targ = Critic(obs_dim, act_dim, hidden=cfg.hidden).to(self.device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q_opt = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=cfg.critic_lr
        )

        self.update_calls = 0 # counts gradient updates (or you can count env steps)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        o = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(o)
        return a.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def hard_update_targets(self):
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        cfg = self.cfg
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device) * cfg.reward_scale
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        # Critic target (NO target actor)
        with torch.no_grad():
            next_acts = self.actor(next_obs) # deterministic actor, current params
            q1_next = self.q1_targ(next_obs, next_acts)
            q2_next = self.q2_targ(next_obs, next_acts)
            q_next = torch.minimum(q1_next, q2_next) # key change vs DDPG
            q_backup = rews + cfg.gamma * (1.0 - done) * q_next

        # Critic loss: both critics regress to same backup
        q1 = self.q1(obs, acts)
        q2 = self.q2(obs, acts)
        critic_loss = 0.5 * ((q1 - q_backup) ** 2).mean() + 0.5 * ((q2 - q_backup) ** 2).mean()

        self.q_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.q_opt.step()

        # Actor update: maximize min(Q1, Q2) under current critics
        pi = self.actor(obs)
        q1_pi = self.q1(obs, pi)
        q2_pi = self.q2(obs, pi)
        actor_loss = -torch.minimum(q1_pi, q2_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # Hard target updates periodically
        self.update_calls += 1
        if self.update_calls % cfg.target_update_period == 0:
            self.hard_update_targets()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }