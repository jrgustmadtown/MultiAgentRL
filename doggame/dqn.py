import os
import sys

import torch
import torch.nn as nn


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from utilities.replay import ReplayBuffer

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
