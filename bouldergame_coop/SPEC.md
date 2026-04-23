# Boulder Cooperative Game - Step 1 Spec

## 1. Overview

Two continuous-control players cooperate to push a boulder through a simple maze into a hole.

- Player 1 can only affect the boulder's x motion when pushing.
- Player 2 can only affect the boulder's y motion when pushing.
- Both players can move in 16 directions (plus optional stay action).
- Boulder is blocked by walls and world boundaries.
- Episode ends on success or horizon.

This is a cooperative game with role-weighted rewards.

## 2. Coordinate System and Bounds

Continuous 2D world in [0, 1] x [0, 1].

- Player positions: p1 = (x1, y1), p2 = (x2, y2)
- Boulder center: b = (bx, by)
- Hole center: h = (hx, hy)
- Boundary is hard-clamped (no position leaves [0, 1]).

## 3. State

Proposed state vector:

s = [x1, y1, x2, y2, bx, by, hx, hy]

Notes:
- Hole can be fixed per episode preset, but remains included in state for extensibility.
- Maze walls are part of environment configuration (not included directly in state vector).

## 4. Action Space

Each player has 17 actions:

- 0: Stay
- 1..16: move direction at 22.5-degree increments around the circle (same style as doggame)

Movement update for each player:

p_i' = clamp(p_i + step_size * dir(action_i))

where clamp enforces world bounds.

## 5. Maze and Obstacles

Maze is represented as axis-aligned wall rectangles.

- Player and boulder movement are both collision-checked against wall rectangles.
- If movement would intersect a wall:
  - for players: clip to nearest valid location or cancel move component (implementation choice, must be consistent)
  - for boulder: stop at collision boundary (no penetration)

Initial preset recommendation:
- A simple corridor-like layout forcing both horizontal and vertical repositioning.

## 6. Push Mechanics

### 6.1 Contact Rule
A player attempts to push if their move segment intersects or enters the boulder contact radius.

Let boulder radius be r_b and player contact radius be r_p.
Contact threshold = r_b + r_p.

### 6.2 Intended Displacement
For each player i:

delta_i = p_i' - p_i

### 6.3 Axis-Limited Influence
- Player 1 contributes only x push: push_x from delta_1.x
- Player 2 contributes only y push: push_y from delta_2.y

Wrong-axis components are ignored.

Examples:
- Player 1 moving mostly vertical into boulder does not create y push.
- Player 2 moving mostly horizontal into boulder does not create x push.

### 6.4 Combined Boulder Update
Proposed one-step candidate update:

b_candidate = (bx + k_push * push_x, by + k_push * push_y)

Then apply collision/boundary resolution:
- Clip to [0, 1]
- Sweep/collision check against maze walls
- Stop at first collision boundary

If no valid movement due to wall, boulder remains at current position for blocked component(s).

## 7. Transition Order (Per Step)

1. Decode both player actions
2. Compute player proposed positions and resolve player-wall/boundary collisions
3. Determine contacts with boulder
4. Compute axis-limited push contributions
5. Compute boulder candidate motion
6. Resolve boulder wall/boundary collisions (stop at wall)
7. Commit next state
8. Compute rewards and done flag

## 8. Reward Design (Cooperative, Role-Weighted)

Define distances at t and t+1:

- dx_t = abs(bx_t - hx)
- dy_t = abs(by_t - hy)
- d_t = sqrt(dx_t^2 + dy_t^2)

Progress terms:

- p_global = d_t - d_{t+1}
- p_x = dx_t - dx_{t+1}
- p_y = dy_t - dy_{t+1}

Rewards:

- r1 = alpha * p_global + beta * p_x + step_penalty
- r2 = alpha * p_global + beta * p_y + step_penalty

Terminal bonus on success:

if success:
- r1 += success_bonus
- r2 += success_bonus

Recommended defaults:
- alpha = 1.0
- beta = 0.5
- step_penalty = -0.01
- success_bonus = 5.0

## 9. Termination

Episode ends if either:

1. Success: boulder reaches hole radius
   - success when distance(b, h) <= r_success
2. Horizon reached: t >= horizon

Recommended default:
- horizon = 30
- r_success = 0.05

## 10. Initial Configuration Defaults

- step_size = 0.08 to 0.12 (start at 0.10)
- k_push = 1.0 (start simple)
- boulder radius r_b = 0.04
- player contact radius r_p = 0.03
- hole radius r_h = 0.05

Spawn defaults:
- boulder starts away from hole, not in collision
- players start near (but not overlapping) boulder

## 11. Learning Setup (Phase 1)

Use the same two-network pattern as existing projects:

- Two Q-networks (one per player)
- Shared cooperative structure but different role-weighted rewards
- Joint action modeling over (a1, a2)
- Replay buffer and target networks

This keeps implementation close to current codebase patterns.

## 12. Deliverables for Next Step

Step 2 will scaffold:

- bouldergame_coop/
  - bouldergame.py (entrypoint)
  - config.py
  - environment.py
  - dqn.py
  - trainer.py
  - policy.py
  - visualization.py
  - io_utils.py
  - CLI.md
  - requirements.txt (if needed)

No implementation done yet in this step beyond this specification.
