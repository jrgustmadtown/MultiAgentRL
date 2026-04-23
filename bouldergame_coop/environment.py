import random

import numpy as np

try:
    from .config import (
        ACTION_DIRS,
        ALPHA_GLOBAL,
        BETA_AXIS,
        BOULDER_RADIUS,
        DEFAULT_BOULDER_START,
        DEFAULT_HOLE,
        DEFAULT_P1_START,
        DEFAULT_P2_START,
        PLAYER_RADIUS,
        DEFAULT_WALLS,
        HOLE_RADIUS,
        PUSH_SCALE,
        STEP_PENALTY,
        STEP_SIZE,
        SUCCESS_BONUS,
        SUCCESS_RADIUS,
        WORLD_MAX,
        WORLD_MIN,
    )
except ImportError:
    from config import (
        ACTION_DIRS,
        ALPHA_GLOBAL,
        BETA_AXIS,
        BOULDER_RADIUS,
        DEFAULT_BOULDER_START,
        DEFAULT_HOLE,
        DEFAULT_P1_START,
        DEFAULT_P2_START,
        PLAYER_RADIUS,
        DEFAULT_WALLS,
        HOLE_RADIUS,
        PUSH_SCALE,
        STEP_PENALTY,
        STEP_SIZE,
        SUCCESS_BONUS,
        SUCCESS_RADIUS,
        WORLD_MAX,
        WORLD_MIN,
    )


class BoulderGame:
    """Environment scaffold for the cooperative boulder game.

    Step 3 will implement full transition and collision logic from SPEC.md.
    """

    def __init__(
        self,
        step_size=STEP_SIZE,
        push_scale=PUSH_SCALE,
        success_radius=SUCCESS_RADIUS,
        p1_start=DEFAULT_P1_START,
        p2_start=DEFAULT_P2_START,
        boulder_start=DEFAULT_BOULDER_START,
        hole=DEFAULT_HOLE,
        walls=None,
    ):
        self.step_size = step_size
        self.push_scale = push_scale
        self.success_radius = success_radius
        self.alpha_global = ALPHA_GLOBAL
        self.beta_axis = BETA_AXIS
        self.step_penalty = STEP_PENALTY
        self.success_bonus = SUCCESS_BONUS
        self.p1_start = p1_start
        self.p2_start = p2_start
        self.boulder_start = boulder_start
        self.hole = hole
        self.walls = DEFAULT_WALLS if walls is None else walls
        self.boulder_radius = BOULDER_RADIUS
        self.hole_radius = HOLE_RADIUS
        self.player_radius = PLAYER_RADIUS
        self.state = self.reset()

    @staticmethod
    def _segment_point_distance(p0, p1, p):
        """Minimum distance from point p to segment p0->p1."""
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        p = np.asarray(p, dtype=float)
        seg = p1 - p0
        seg_norm2 = np.dot(seg, seg)
        if seg_norm2 <= 1e-12:
            return float(np.linalg.norm(p - p0))
        t = float(np.dot(p - p0, seg) / seg_norm2)
        t = max(0.0, min(1.0, t))
        proj = p0 + t * seg
        return float(np.linalg.norm(p - proj))

    def _clamp_to_bounds(self, x, y, radius):
        min_c = WORLD_MIN + radius
        max_c = WORLD_MAX - radius
        return (float(np.clip(x, min_c, max_c)), float(np.clip(y, min_c, max_c)))

    def _in_expanded_rect(self, x, y, rect, radius):
        x0, y0, x1, y1 = rect
        return (x0 - radius) <= x <= (x1 + radius) and (y0 - radius) <= y <= (y1 + radius)

    @staticmethod
    def _point_in_rect(x, y, rect):
        x0, y0, x1, y1 = rect
        return x0 <= x <= x1 and y0 <= y <= y1

    @staticmethod
    def _rects_overlap(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return (ax0 <= bx1 and ax1 >= bx0 and ay0 <= by1 and ay1 >= by0)

    def _segment_intersects_rect(self, p0, p1, rect):
        """Approximate segment-rectangle intersection by dense sampling."""
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        seg_len = float(np.linalg.norm(p1 - p0))
        samples = max(8, int(np.ceil(seg_len / 0.005)))
        for k in range(samples + 1):
            t = k / samples
            px, py = p0 + t * (p1 - p0)
            if self._point_in_rect(float(px), float(py), rect):
                return True
        return False

    def _boulder_rect(self, bx, by, extra=0.0):
        r = self.boulder_radius + extra
        return (bx - r, by - r, bx + r, by + r)

    def _hole_rect(self, hx, hy):
        r = self.hole_radius
        return (hx - r, hy - r, hx + r, hy + r)

    def _collides_walls(self, x, y, radius):
        for rect in self.walls:
            if self._in_expanded_rect(x, y, rect, radius):
                return True
        return False

    def _move_with_collisions(self, start_xy, delta_xy, radius):
        """Move from start by delta, stopping at first collision and respecting bounds."""
        sx, sy = float(start_xy[0]), float(start_xy[1])
        dx, dy = float(delta_xy[0]), float(delta_xy[1])

        step_len = float(np.linalg.norm([dx, dy]))
        if step_len <= 1e-12:
            return (sx, sy)

        substeps = max(1, int(np.ceil(step_len / 0.005)))
        last_valid = (sx, sy)
        for k in range(1, substeps + 1):
            t = k / substeps
            cx = sx + dx * t
            cy = sy + dy * t
            cx, cy = self._clamp_to_bounds(cx, cy, radius)
            if self._collides_walls(cx, cy, radius):
                break
            last_valid = (cx, cy)
        return last_valid

    def _action_delta(self, action, step_size):
        ax, ay = ACTION_DIRS[action]
        return (step_size * ax, step_size * ay)

    def reset(self):
        self.state = (
            float(self.p1_start[0]),
            float(self.p1_start[1]),
            float(self.p2_start[0]),
            float(self.p2_start[1]),
            float(self.boulder_start[0]),
            float(self.boulder_start[1]),
            float(self.hole[0]),
            float(self.hole[1]),
        )
        return self.state

    def sample_state(self):
        # Sample valid positions (not colliding with walls for players/boulder).
        def sample_xy(radius):
            for _ in range(200):
                x = random.uniform(WORLD_MIN + radius, WORLD_MAX - radius)
                y = random.uniform(WORLD_MIN + radius, WORLD_MAX - radius)
                if not self._collides_walls(x, y, radius):
                    return (x, y)
            return self._clamp_to_bounds(0.5, 0.5, radius)

        p1 = sample_xy(self.player_radius)
        p2 = sample_xy(self.player_radius)
        b = sample_xy(self.boulder_radius)

        return (
            float(p1[0]),
            float(p1[1]),
            float(p2[0]),
            float(p2[1]),
            float(b[0]),
            float(b[1]),
            float(self.hole[0]),
            float(self.hole[1]),
        )

    def transition(self, s, a1, a2):
        x1, y1, x2, y2, bx, by, hx, hy = s

        # 1) Player movement with wall/boundary collisions.
        d1 = self._action_delta(a1, self.step_size)
        d2 = self._action_delta(a2, self.step_size)

        p1_next = self._move_with_collisions((x1, y1), d1, self.player_radius)
        p2_next = self._move_with_collisions((x2, y2), d2, self.player_radius)

        # 2) If inside the boulder square, each player can move one axis by fixed step.
        # Player 1 controls x-axis motion, Player 2 controls y-axis motion.
        boulder_rect = self._boulder_rect(bx, by)
        p1_inside = (
            self._point_in_rect(x1, y1, boulder_rect)
            or self._point_in_rect(p1_next[0], p1_next[1], boulder_rect)
            or self._segment_intersects_rect((x1, y1), p1_next, boulder_rect)
        )
        p2_inside = (
            self._point_in_rect(x2, y2, boulder_rect)
            or self._point_in_rect(p2_next[0], p2_next[1], boulder_rect)
            or self._segment_intersects_rect((x2, y2), p2_next, boulder_rect)
        )

        a1_x, _ = ACTION_DIRS[a1]
        _, a2_y = ACTION_DIRS[a2]
        move_x = self.step_size * a1_x if p1_inside else 0.0
        move_y = self.step_size * a2_y if p2_inside else 0.0

        # 3) Move boulder with wall/boundary collision stopping.
        b_next = self._move_with_collisions((bx, by), (move_x, move_y), self.boulder_radius)

        return (
            float(p1_next[0]),
            float(p1_next[1]),
            float(p2_next[0]),
            float(p2_next[1]),
            float(b_next[0]),
            float(b_next[1]),
            float(hx),
            float(hy),
        )

    def reward(self, s, a1, a2):
        s_next = self.transition(s, a1, a2)

        bx_t, by_t = s[4], s[5]
        bx_n, by_n = s_next[4], s_next[5]
        hx, hy = s[6], s[7]

        dx_t = abs(bx_t - hx)
        dy_t = abs(by_t - hy)
        d_t = float(np.sqrt(dx_t ** 2 + dy_t ** 2))

        dx_n = abs(bx_n - hx)
        dy_n = abs(by_n - hy)
        d_n = float(np.sqrt(dx_n ** 2 + dy_n ** 2))

        p_global = d_t - d_n
        p_x = dx_t - dx_n
        p_y = dy_t - dy_n

        r1 = self.alpha_global * p_global + self.beta_axis * p_x + self.step_penalty
        r2 = self.alpha_global * p_global + self.beta_axis * p_y + self.step_penalty

        if self.is_success(s_next):
            r1 += self.success_bonus
            r2 += self.success_bonus

        return float(r1), float(r2)

    def is_success(self, s):
        bx, by = s[4], s[5]
        hx, hy = s[6], s[7]
        boulder_rect = self._boulder_rect(bx, by)
        hole_rect = self._hole_rect(hx, hy)
        return self._rects_overlap(boulder_rect, hole_rect)


def parse_position(s):
    x, y = s.split(",")
    return (float(x), float(y))
