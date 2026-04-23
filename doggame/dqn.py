from collections import deque
import random

import torch
import torch.nn as nn

try:
    from .config import NUM_ACTIONS, device
except ImportError:
    from config import NUM_ACTIONS, device


def encode_state(s):
    """State s = (p1_x, p1_y, p2_x, p2_y) in [0,1] space."""
    return torch.tensor(s, dtype=torch.float32, device=device)


class DQN(nn.Module):
    """Neural network that approximates Q(s, a1, a2)."""

    def __init__(self, state_dim=4, action_dim=None):
        super().__init__()
        if action_dim is None:
            action_dim = NUM_ACTIONS * NUM_ACTIONS
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
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
