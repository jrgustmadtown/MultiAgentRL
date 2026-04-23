import os
import sys

import torch
import torch.nn as nn


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from utilities.replay import ReplayBuffer

try:
    from .config import NUM_JOINT_ACTIONS, device
except ImportError:
    from config import NUM_JOINT_ACTIONS, device


def encode_state(state):
    return torch.tensor(state, dtype=torch.float32, device=device)


class DQN(nn.Module):
    """Joint-action Q-network Q(s, a1, a2)."""

    def __init__(self, state_dim=8, action_dim=NUM_JOINT_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, s):
        return self.net(s)
