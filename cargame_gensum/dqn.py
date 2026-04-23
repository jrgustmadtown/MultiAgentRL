import os
import sys

import torch
import torch.nn as nn


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from utilities.replay import ReplayBuffer

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
