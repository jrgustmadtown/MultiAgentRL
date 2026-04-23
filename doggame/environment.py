import random

import numpy as np

try:
    from .config import ACTION_DIRS, DEFAULT_HOUSE1, DEFAULT_HOUSE2
except ImportError:
    from config import ACTION_DIRS, DEFAULT_HOUSE1, DEFAULT_HOUSE2


def dog_position(s):
    """Dog is at the midpoint of the two players."""
    return ((s[0] + s[2]) / 2, (s[1] + s[3]) / 2)


def distance(p1, p2):
    """Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class DogGame:
    """
    Dog Game Environment.

    State: (p1_x, p1_y, p2_x, p2_y) - all in [0, 1]
    Dog: midpoint of players
    Reward: -distance(dog, my_house)
    """

    def __init__(self, step_size=0.1, house1=None, house2=None):
        self.step_size = step_size
        self.house1 = house1 if house1 else DEFAULT_HOUSE1
        self.house2 = house2 if house2 else DEFAULT_HOUSE2

    def move(self, x, y, action):
        """Apply action and clamp to [0, 1]."""
        dx, dy = ACTION_DIRS[action]
        x_new = np.clip(x + dx * self.step_size, 0, 1)
        y_new = np.clip(y + dy * self.step_size, 0, 1)
        return x_new, y_new

    def transition(self, s, a1, a2):
        """Return next state after both players move."""
        x1_new, y1_new = self.move(s[0], s[1], a1)
        x2_new, y2_new = self.move(s[2], s[3], a2)
        return (x1_new, y1_new, x2_new, y2_new)

    def reward(self, s, a1, a2):
        """Compute rewards for both players."""
        s_next = self.transition(s, a1, a2)
        dog = dog_position(s_next)
        r1 = -distance(dog, self.house1)
        r2 = -distance(dog, self.house2)
        return r1, r2

    def sample_state(self):
        """
        Sample a random state with bias toward corners/boundaries.
        50% uniform random, 50% near boundaries.
        """
        if random.random() < 0.5:
            return (random.random(), random.random(), random.random(), random.random())

        def biased_coord():
            if random.random() < 0.5:
                return random.betavariate(0.3, 2)
            return random.betavariate(2, 0.3)

        return (biased_coord(), biased_coord(), biased_coord(), biased_coord())
