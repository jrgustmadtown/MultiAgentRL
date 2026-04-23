import os
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from utilities.runtime import is_headless_matplotlib

try:
    from .environment import distance, dog_position
    from .io_utils import output_path
    from .policy import rollout
except ImportError:
    from environment import distance, dog_position
    from io_utils import output_path
    from policy import rollout


def draw_trajectory(ax, traj, house, wall=None):
    """Draw rollout with player/dog step arrows and house square."""
    ax.clear()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")

    hx, hy = house
    ax.plot(hx, hy, "s", color="gold", markersize=15, alpha=0.7)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "k-", linewidth=2)
    if wall is not None:
        (wx0, wy0), (wx1, wy1) = wall
        ax.plot([wx0, wx1], [wy0, wy1], color="black", linewidth=4, alpha=0.7)

    total_steps = len(traj) - 1

    for i, (s, s_next) in enumerate(zip(traj[:-1], traj[1:])):
        alpha = 0.3 + 0.7 * (i / max(1, total_steps))

        ax.arrow(
            s[0],
            s[1],
            s_next[0] - s[0],
            s_next[1] - s[1],
            color="red",
            alpha=alpha,
            head_width=0.02,
            length_includes_head=True,
            linewidth=1.5,
        )
        ax.arrow(
            s[2],
            s[3],
            s_next[2] - s[2],
            s_next[3] - s[3],
            color="blue",
            alpha=alpha,
            head_width=0.02,
            length_includes_head=True,
            linewidth=1.5,
        )

    dog_traj = [dog_position(s) for s in traj]
    for i, (d, d_next) in enumerate(zip(dog_traj[:-1], dog_traj[1:])):
        alpha = 0.3 + 0.7 * (i / max(1, total_steps))
        ax.arrow(
            d[0],
            d[1],
            d_next[0] - d[0],
            d_next[1] - d[1],
            color="green",
            alpha=alpha,
            head_width=0.015,
            length_includes_head=True,
            linewidth=1,
            linestyle="--",
        )

    final = traj[-1]
    ax.plot(final[0], final[1], "o", color="red", markersize=8)
    ax.plot(final[2], final[3], "o", color="blue", markersize=8)
    final_dog = dog_position(final)
    ax.plot(final_dog[0], final_dog[1], "o", color="green", markersize=10)

    start = traj[0]
    ax.plot(start[0], start[1], "x", color="darkred", markersize=8)
    ax.plot(start[2], start[3], "x", color="darkblue", markersize=8)


def plot_training_losses(losses1, losses2):
    plt.figure()
    plt.plot(losses1, label="P1 Loss", alpha=0.7)
    plt.plot(losses2, label="P2 Loss", alpha=0.7)
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Dog Game Coop - Training Loss")
    plt.legend()
    loss_path = output_path("planning_loss.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"Saved {loss_path}")


def run_rollout_visualization(env, policy_fn, house, horizon):
    is_headless = is_headless_matplotlib(plt)

    fig, ax = plt.subplots(figsize=(8, 8))

    if is_headless:
        print("Headless mode. Saving 5 rollouts...")
        for i in range(5):
            s0 = env.sample_state()
            traj = rollout(env, s0, policy_fn, horizon)
            draw_trajectory(ax, traj, house, wall=env.wall)

            final_dog = dog_position(traj[-1])
            d = distance(final_dog, house)

            ax.set_title(f"Rollout {i + 1}: Dog to house={d:.3f} | Steps={len(traj)-1}")
            rollout_file = output_path(f"rollout_{i + 1}.png")
            fig.savefig(rollout_file, dpi=150, bbox_inches="tight")
            print(
                f"  Saved {rollout_file} | Start: "
                f"({s0[0]:.2f},{s0[1]:.2f}), ({s0[2]:.2f},{s0[3]:.2f})"
            )
        return

    from matplotlib.widgets import Button

    current_state = [env.sample_state()]

    def update_plot():
        s0 = current_state[0]
        traj = rollout(env, s0, policy_fn, horizon)
        draw_trajectory(ax, traj, house, wall=env.wall)

        final_dog = dog_position(traj[-1])
        d = distance(final_dog, house)

        fig.suptitle("Dog Game Cooperative Rollout", fontsize=14)
        ax.set_title(f"Dog to House: {d:.3f} | Steps: {len(traj) - 1}", fontsize=10)
        fig.canvas.draw_idle()

    plt.subplots_adjust(bottom=0.15)
    ax_new = plt.axes([0.4, 0.02, 0.2, 0.05])
    btn = Button(ax_new, "New Random")

    def new_state(_event):
        current_state[0] = env.sample_state()
        update_plot()

    btn.on_clicked(new_state)
    update_plot()
    plt.show()
