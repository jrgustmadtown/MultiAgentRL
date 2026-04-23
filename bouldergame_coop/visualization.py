import os
import sys
import random

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from utilities.runtime import is_headless_matplotlib

try:
    from .config import HOLE_RADIUS, PLAYER_RADIUS
    from .policy import rollout
except ImportError:
    from config import HOLE_RADIUS, PLAYER_RADIUS
    from policy import rollout


def draw_trajectory(ax, traj, walls, hole, boulder_radius, hole_radius):
    """Draw a trajectory with maze walls, players, boulder path, and hole."""
    ax.clear()
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    for (x0, y0, x1, y1) in walls:
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="black", alpha=0.25)
        ax.add_patch(rect)

    hx, hy = hole
    hole_patch = plt.Rectangle(
        (hx - hole_radius, hy - hole_radius),
        2 * hole_radius,
        2 * hole_radius,
        color="gold",
        alpha=0.6,
    )
    ax.add_patch(hole_patch)

    p1_pts = np.array([(s[0], s[1]) for s in traj], dtype=float)
    p2_pts = np.array([(s[2], s[3]) for s in traj], dtype=float)
    b_pts = np.array([(s[4], s[5]) for s in traj], dtype=float)

    total_steps = len(traj) - 1

    for i, (s, s_next) in enumerate(zip(traj[:-1], traj[1:])):
        if i == total_steps - 1:
            alpha = 1.0
        else:
            alpha = 0.2 + 0.6 * (i / max(1, total_steps))

        p1_dx = s_next[0] - s[0]
        p1_dy = s_next[1] - s[1]
        if abs(p1_dx) > 1e-8 or abs(p1_dy) > 1e-8:
            ax.arrow(
                s[0],
                s[1],
                p1_dx,
                p1_dy,
                color="red",
                alpha=alpha,
                head_width=0.015,
                length_includes_head=True,
                linewidth=1.5,
            )

        p2_dx = s_next[2] - s[2]
        p2_dy = s_next[3] - s[3]
        if abs(p2_dx) > 1e-8 or abs(p2_dy) > 1e-8:
            ax.arrow(
                s[2],
                s[3],
                p2_dx,
                p2_dy,
                color="blue",
                alpha=alpha,
                head_width=0.015,
                length_includes_head=True,
                linewidth=1.5,
            )

        b_dx = s_next[4] - s[4]
        b_dy = s_next[5] - s[5]
        if abs(b_dx) > 1e-8 or abs(b_dy) > 1e-8:
            ax.arrow(
                s[4],
                s[5],
                b_dx,
                b_dy,
                color="green",
                alpha=alpha,
                head_width=0.012,
                length_includes_head=True,
                linewidth=1.2,
                linestyle="--",
            )

    ax.plot(p1_pts[:, 0], p1_pts[:, 1], color="red", alpha=0.6, linewidth=1.5, label="Player 1")
    ax.plot(p2_pts[:, 0], p2_pts[:, 1], color="blue", alpha=0.6, linewidth=1.5, label="Player 2")
    ax.plot(b_pts[:, 0], b_pts[:, 1], color="green", alpha=0.8, linewidth=2.0, label="Boulder")

    ax.scatter(p1_pts[0, 0], p1_pts[0, 1], color="darkred", marker="x", s=60)
    ax.scatter(p2_pts[0, 0], p2_pts[0, 1], color="darkblue", marker="x", s=60)
    ax.scatter(b_pts[0, 0], b_pts[0, 1], color="darkgreen", marker="x", s=60)

    boulder_patch = plt.Rectangle(
        (b_pts[-1, 0] - boulder_radius, b_pts[-1, 1] - boulder_radius),
        2 * boulder_radius,
        2 * boulder_radius,
        color="green",
        alpha=0.35,
    )
    ax.add_patch(boulder_patch)

    ax.scatter(p1_pts[-1, 0], p1_pts[-1, 1], color="red", s=30)
    ax.scatter(p2_pts[-1, 0], p2_pts[-1, 1], color="blue", s=30)
    ax.scatter(b_pts[-1, 0], b_pts[-1, 1], color="green", s=40)

    ax.legend(loc="upper left", fontsize=8)


def draw_vector_field(_nets, _env):
    """Optional for this game; not implemented in Step 6."""
    return None


def plot_training_losses(losses1, losses2, output_path):
    plt.figure()
    plt.plot(losses1, label="P1 Loss", alpha=0.75)
    plt.plot(losses2, label="P2 Loss", alpha=0.75)
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Boulder Game - Training Loss")
    plt.legend()
    loss_file = output_path("planning_loss.png")
    plt.savefig(loss_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {loss_file}")


def run_rollout_visualization(env, policy_fn, horizon, output_path, use_random_starts=False):
    is_headless = is_headless_matplotlib(plt)

    fig, ax = plt.subplots(figsize=(8, 8))

    if is_headless:
        print("Headless mode. Saving 5 rollouts...")
        for i in range(5):
            s0 = env.sample_state() if use_random_starts else env.reset()
            traj = rollout(env, s0, policy_fn, horizon)
            draw_trajectory(ax, traj, env.walls, env.hole, env.boulder_radius, HOLE_RADIUS)

            bx, by = traj[-1][4], traj[-1][5]
            hx, hy = env.hole
            dist = float(np.sqrt((bx - hx) ** 2 + (by - hy) ** 2))
            ax.set_title(f"Rollout {i + 1}: dist(boulder,hole)={dist:.3f}")

            rollout_file = output_path(f"rollout_{i + 1}.png")
            fig.savefig(rollout_file, dpi=150, bbox_inches="tight")
            print(f"  Saved {rollout_file}")
        return

    from matplotlib.widgets import Button

    current_state = [env.sample_state() if use_random_starts else env.reset()]

    def update_plot():
        s0 = current_state[0]
        traj = rollout(env, s0, policy_fn, horizon)
        draw_trajectory(ax, traj, env.walls, env.hole, env.boulder_radius, HOLE_RADIUS)

        bx, by = traj[-1][4], traj[-1][5]
        hx, hy = env.hole
        dist = float(np.sqrt((bx - hx) ** 2 + (by - hy) ** 2))

        fig.suptitle("Boulder Cooperative Rollout", fontsize=14)
        ax.set_title(f"Distance to hole: {dist:.3f} | Steps: {len(traj) - 1}", fontsize=10)
        fig.canvas.draw_idle()

    plt.subplots_adjust(bottom=0.15)
    ax_new = plt.axes([0.4, 0.02, 0.2, 0.05])
    btn = Button(ax_new, "New Random")

    def new_state(_event):
        current_state[0] = env.sample_state() if use_random_starts else env.reset()
        update_plot()

    btn.on_clicked(new_state)
    update_plot()
    plt.show()
