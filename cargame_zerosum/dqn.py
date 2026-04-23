from collections import deque
import random

import torch
import torch.nn as nn

try:
    from .config import device
except ImportError:
    from config import device


def encode_state(s, grid_size):
    """
    Normalizes state coordinates for the Neural Network.
    Positions are in [0, 1-STEP_SIZE] space, where each grid cell = STEP_SIZE.
    """
    step = 1 / grid_size
    return torch.tensor(
        [s[0] * step, s[1] * step, s[2] * step, s[3] * step],
        dtype=torch.float32,
        device=device,
    )


class DQN(nn.Module):
    """Neural Network that approximates Q(s, a1, a2)."""

    def __init__(self, state_dim=4, action_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, s):
        return self.net(s)


class ReplayBuffer:
    """Fixed-size buffer to store (state_tensor, target_q) tuples."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_tensor, target_q):
        self.buffer.append((state_tensor, target_q))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = torch.stack([x[0] for x in batch])
        targets = torch.stack([x[1] for x in batch])
        return states, targets

    def __len__(self):
        return len(self.buffer)
