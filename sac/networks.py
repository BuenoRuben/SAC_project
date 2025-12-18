import math
import torch
import torch.nn as nn


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, hidden, hidden, 1])

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, 1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 256,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.backbone = mlp([obs_dim, hidden, hidden, hidden], output_activation=nn.ReLU)
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs: torch.Tensor):
        h = self.backbone(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, obs: torch.Tensor):
        mu, log_std = self(obs)
        std = torch.exp(log_std)

        eps = torch.randn_like(mu)
        u = mu + eps * std
        a = torch.tanh(u)

        logp_u = -0.5 * (((u - mu) / (std + 1e-8)) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi))
        logp_u = logp_u.sum(dim=-1, keepdim=True)

        correction = torch.log(1.0 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        logp_a = logp_u - correction

        mu_tanh = torch.tanh(mu)
        return a, logp_a, mu_tanh
