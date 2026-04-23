import random

import numpy as np

try:
    from .config import (
        ACTION_DIRS,
        DEFAULT_HOUSE,
        DEFAULT_WALL,
        STEP_SIZE,
        SUCCESS_BONUS,
        SUCCESS_RADIUS,
    )
except ImportError:
    from config import ACTION_DIRS, DEFAULT_HOUSE, DEFAULT_WALL, STEP_SIZE, SUCCESS_BONUS, SUCCESS_RADIUS


def dog_position(s):
    """Dog is at the midpoint of the two players."""
    return ((s[0] + s[2]) / 2.0, (s[1] + s[3]) / 2.0)


def distance(p1, p2):
    """Euclidean distance between two points."""
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


class DogGameCoop:
    """
    Cooperative dog game.

    State: (p1_x, p1_y, p2_x, p2_y), each in [0, 1]
    Dog: midpoint of players
    Shared objective: move dog to a single house target
    """

    def __init__(
        self,
        step_size=STEP_SIZE,
        house=None,
        wall=None,
        success_radius=SUCCESS_RADIUS,
        success_bonus=SUCCESS_BONUS,
    ):
        self.step_size = step_size
        self.house = house if house else DEFAULT_HOUSE
        self.wall = wall if wall else DEFAULT_WALL
        self.success_radius = success_radius
        self.success_bonus = success_bonus

    @staticmethod
    def _orientation(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    @staticmethod
    def _on_segment(a, b, c):
        return (
            min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
            and min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9
        )

    @classmethod
    def _segments_intersect(cls, p1, p2, q1, q2):
        o1 = cls._orientation(p1, p2, q1)
        o2 = cls._orientation(p1, p2, q2)
        o3 = cls._orientation(q1, q2, p1)
        o4 = cls._orientation(q1, q2, p2)

        if o1 * o2 < 0 and o3 * o4 < 0:
            return True
        if abs(o1) <= 1e-9 and cls._on_segment(p1, q1, p2):
            return True
        if abs(o2) <= 1e-9 and cls._on_segment(p1, q2, p2):
            return True
        if abs(o3) <= 1e-9 and cls._on_segment(q1, p1, q2):
            return True
        if abs(o4) <= 1e-9 and cls._on_segment(q1, p2, q2):
            return True
        return False

    def _blocked_by_wall(self, start_xy, end_xy):
        wall_start, wall_end = self.wall
        return self._segments_intersect(start_xy, end_xy, wall_start, wall_end)

    def _slide_midpoint_along_wall(self, s, s_next):
        """Remove blocked midpoint motion normal to the wall and keep tangent motion."""
        dog_before = np.array(dog_position(s), dtype=float)
        dog_after = np.array(dog_position(s_next), dtype=float)
        dog_delta = dog_after - dog_before

        wall_start = np.array(self.wall[0], dtype=float)
        wall_end = np.array(self.wall[1], dtype=float)
        wall_vec = wall_end - wall_start
        wall_norm = float(np.linalg.norm(wall_vec))
        if wall_norm <= 1e-12:
            return s

        wall_tangent = wall_vec / wall_norm
        slide_delta = float(np.dot(dog_delta, wall_tangent)) * wall_tangent
        blocked_delta = dog_delta - slide_delta

        if float(np.linalg.norm(blocked_delta)) <= 1e-12:
            return s_next

        p1 = np.array([s_next[0], s_next[1]], dtype=float) - blocked_delta
        p2 = np.array([s_next[2], s_next[3]], dtype=float) - blocked_delta
        p1 = np.clip(p1, 0.0, 1.0)
        p2 = np.clip(p2, 0.0, 1.0)

        candidate = (float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]))

        if self._blocked_by_wall((s[0], s[1]), (candidate[0], candidate[1])):
            return s
        if self._blocked_by_wall((s[2], s[3]), (candidate[2], candidate[3])):
            return s
        if self._blocked_by_wall(dog_position(s), dog_position(candidate)):
            return s
        return candidate

    def move(self, x, y, action):
        """Apply action and clamp to [0, 1]."""
        dx, dy = ACTION_DIRS[action]
        x_new = float(np.clip(x + dx * self.step_size, 0.0, 1.0))
        y_new = float(np.clip(y + dy * self.step_size, 0.0, 1.0))
        if self._blocked_by_wall((x, y), (x_new, y_new)):
            return float(x), float(y)
        return x_new, y_new

    def transition(self, s, a1, a2):
        """Return next state after both players move."""
        dog_before = dog_position(s)
        x1_new, y1_new = self.move(s[0], s[1], a1)
        x2_new, y2_new = self.move(s[2], s[3], a2)
        s_next = (x1_new, y1_new, x2_new, y2_new)
        dog_after = dog_position(s_next)
        if self._blocked_by_wall(dog_before, dog_after):
            return self._slide_midpoint_along_wall(s, s_next)
        return s_next

    def is_success(self, s):
        dog = dog_position(s)
        return distance(dog, self.house) <= self.success_radius

    def reward(self, s, a1, a2):
        """Shared reward: negative dog-house distance (+ optional terminal bonus)."""
        s_next = self.transition(s, a1, a2)
        dog = dog_position(s_next)
        r = -distance(dog, self.house)
        if self.is_success(s_next):
            r += self.success_bonus
        return float(r), float(r)

    def sample_state(self):
        """Sample random states with mild boundary bias, like original doggame."""
        if random.random() < 0.5:
            return (random.random(), random.random(), random.random(), random.random())

        def biased_coord():
            if random.random() < 0.5:
                return random.betavariate(0.3, 2)
            return random.betavariate(2, 0.3)

        return (biased_coord(), biased_coord(), biased_coord(), biased_coord())
