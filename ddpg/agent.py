from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.optim as optim

from .networks import Actor, Critic


@dataclass
class DDPGConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    hidden: int = 256
    batch_size: int = 256
    reward_scale: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DDPGAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: DDPGConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actor = Actor(obs_dim, act_dim, hidden=cfg.hidden).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hidden=cfg.hidden).to(self.device)

        self.actor_targ = Actor(obs_dim, act_dim, hidden=cfg.hidden).to(self.device)
        self.critic_targ = Critic(obs_dim, act_dim, hidden=cfg.hidden).to(self.device)
        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic_targ.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)


    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        o = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(o)
        return a.squeeze(0).cpu().numpy()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        cfg = self.cfg
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device) * cfg.reward_scale
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        # Critic update
        with torch.no_grad():
            next_acts = self.actor_targ(next_obs)
            q_next = self.critic_targ(next_obs, next_acts)
            q_backup = rews + cfg.gamma * (1.0 - done) * q_next

        q = self.critic(obs, acts)
        critic_loss = 0.5 * ((q - q_backup) ** 2).mean()

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        # Maximize Q(s, actor(s)) => minimize -Q(...)
        pi = self.actor(obs)
        actor_loss = -self.critic(obs, pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # Target updates (Polyak)
        with torch.no_grad():
            for p, p_targ in zip(self.actor.parameters(), self.actor_targ.parameters()):
                p_targ.data.mul_(1.0 - cfg.tau)
                p_targ.data.add_(cfg.tau * p.data)

            for p, p_targ in zip(self.critic.parameters(), self.critic_targ.parameters()):
                p_targ.data.mul_(1.0 - cfg.tau)
                p_targ.data.add_(cfg.tau * p.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }
