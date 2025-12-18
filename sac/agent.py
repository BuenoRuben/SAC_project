from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.optim as optim

from .networks import Actor, QNetwork, ValueNetwork


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    hidden: int = 256
    batch_size: int = 256
    reward_scale: float = 5.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: SACConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actor = Actor(obs_dim, act_dim, hidden=cfg.hidden).to(self.device)
        self.q1 = QNetwork(obs_dim, act_dim, hidden=cfg.hidden).to(self.device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden=cfg.hidden).to(self.device)
        self.v = ValueNetwork(obs_dim, hidden=cfg.hidden).to(self.device)
        self.v_targ = ValueNetwork(obs_dim, hidden=cfg.hidden).to(self.device)
        self.v_targ.load_state_dict(self.v.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=cfg.lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=cfg.lr)
        self.v_opt = optim.Adam(self.v.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool) -> np.ndarray:
        o = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _, mu_tanh = self.actor.sample(o)
        out = mu_tanh if deterministic else a
        return out.squeeze(0).cpu().numpy()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        cfg = self.cfg
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device) * cfg.reward_scale
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        # Update V
        with torch.no_grad():
            a_pi, logp_a, _ = self.actor.sample(obs)
            q_pi = torch.min(self.q1(obs, a_pi), self.q2(obs, a_pi))
            v_target = q_pi - logp_a

        v = self.v(obs)
        v_loss = 0.5 * ((v - v_target) ** 2).mean()
        self.v_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_opt.step()

        # Update Q1/Q2
        with torch.no_grad():
            v_next = self.v_targ(next_obs)
            q_backup = rews + cfg.gamma * (1.0 - done) * v_next

        q1 = self.q1(obs, acts)
        q2 = self.q2(obs, acts)
        q1_loss = 0.5 * ((q1 - q_backup) ** 2).mean()
        q2_loss = 0.5 * ((q2 - q_backup) ** 2).mean()

        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.q2_opt.step()

        # Update policy
        a_pi, logp_a, _ = self.actor.sample(obs)
        q_pi = torch.min(self.q1(obs, a_pi), self.q2(obs, a_pi))
        pi_loss = (logp_a - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.actor_opt.step()

        # Target V update (Polyak averaging)
        with torch.no_grad():
            for p, p_targ in zip(self.v.parameters(), self.v_targ.parameters()):
                p_targ.data.mul_(1.0 - cfg.tau)
                p_targ.data.add_(cfg.tau * p.data)

        return {
            "v_loss": float(v_loss.item()),
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "pi_loss": float(pi_loss.item()),
        }
