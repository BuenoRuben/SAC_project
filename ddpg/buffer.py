import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rews = np.zeros((self.capacity, 1), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, o, a, r, o2, d) -> None:
        self.obs[self.ptr] = o
        self.acts[self.ptr] = a
        self.rews[self.ptr] = r
        self.next_obs[self.ptr] = o2
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=torch.from_numpy(self.obs[idx]),
            acts=torch.from_numpy(self.acts[idx]),
            rews=torch.from_numpy(self.rews[idx]),
            next_obs=torch.from_numpy(self.next_obs[idx]),
            done=torch.from_numpy(self.done[idx]),
        )
