import random
from collections import deque

import torch


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
