import numpy as np

try:
    from .config import CRASH_PENALTY, GRID_REWARD_MAX, LIVING_COST, STAY_PENALTY
except ImportError:
    from config import CRASH_PENALTY, GRID_REWARD_MAX, LIVING_COST, STAY_PENALTY


class CarGame:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.states = [
            (x1, y1, x2, y2)
            for x1 in range(grid_size)
            for y1 in range(grid_size)
            for x2 in range(grid_size)
            for y2 in range(grid_size)
        ]
        self.grid_reward = self._make_grid_reward(grid_size)

    def _make_grid_reward(self, n):
        c = (n - 1) / 2
        grid = np.zeros((n, n))
        for x in range(n):
            for y in range(n):
                dist = abs(x - c) + abs(y - c)
                grid[x, y] = GRID_REWARD_MAX * (1 - dist / (2 * c))
        return grid

    def move(self, x, y, a):
        if a == 0:
            y_new = min(self.grid_size - 1, y + 1)
            x_new = x
        elif a == 1:
            y_new = max(0, y - 1)
            x_new = x
        elif a == 2:
            x_new = max(0, x - 1)
            y_new = y
        elif a == 3:
            x_new = min(self.grid_size - 1, x + 1)
            y_new = y
        return x_new, y_new

    def transition(self, s, a1, a2):
        x1n, y1n = self.move(s[0], s[1], a1)
        x2n, y2n = self.move(s[2], s[3], a2)
        return (x1n, y1n, x2n, y2n)

    def reward(self, s, a1, a2):
        x1, y1, x2, y2 = s
        sn = self.transition(s, a1, a2)
        if (sn[0], sn[1]) == (sn[2], sn[3]):
            return CRASH_PENALTY, CRASH_PENALTY
        r1 = self.grid_reward[sn[0], sn[1]] - LIVING_COST
        r2 = self.grid_reward[sn[2], sn[3]] - LIVING_COST
        if (sn[0], sn[1]) == (x1, y1):
            r1 += STAY_PENALTY
        if (sn[2], sn[3]) == (x2, y2):
            r2 += STAY_PENALTY
        return r1, r2
