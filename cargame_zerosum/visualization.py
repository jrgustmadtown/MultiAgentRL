import random
import os

import matplotlib.pyplot as plt
import numpy as np

try:
    from .policy import rollout
except ImportError:
    from policy import rollout


def draw_trajectory(ax, traj, grid_size, subtitle=""):
    ax.clear()
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(range(grid_size + 1))
    ax.set_yticks(range(grid_size + 1))
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_title(subtitle, fontsize=10, color="gray", pad=6)

    offset = 0.12
    off1 = np.array([offset, offset])
    off2 = np.array([-offset, -offset])

    total_steps = len(traj) - 1

    for i, (s, s_next) in enumerate(zip(traj[:-1], traj[1:])):
        if i == total_steps - 1:
            alpha = 1.0
        else:
            alpha = 0.2 + 0.6 * (i / total_steps)

        p1_start = np.array([s[0] + 0.5, s[1] + 0.5]) + off1
        p1_end = np.array([s_next[0] + 0.5, s_next[1] + 0.5]) + off1
        d1 = p1_end - p1_start

        if np.linalg.norm(d1) > 1e-6:
            ax.arrow(
                p1_start[0],
                p1_start[1],
                d1[0],
                d1[1],
                color="red",
                alpha=alpha,
                head_width=0.15,
                length_includes_head=True,
            )

        p2_start = np.array([s[2] + 0.5, s[3] + 0.5]) + off2
        p2_end = np.array([s_next[2] + 0.5, s_next[3] + 0.5]) + off2
        d2 = p2_end - p2_start

        if np.linalg.norm(d2) > 1e-6:
            ax.arrow(
                p2_start[0],
                p2_start[1],
                d2[0],
                d2[1],
                color="blue",
                alpha=alpha,
                head_width=0.15,
                length_includes_head=True,
            )


def run_rollout_visualization(env, policy, grid_size, output_path):
    valid_states = [s for s in env.states if (s[0], s[1]) != (s[2], s[3])]

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    is_headless = plt.get_backend().lower() == "agg" or not has_display

    fig, ax = plt.subplots()
    idx = [0]

    def update_plot(_event=None):
        s0 = valid_states[idx[0]]
        traj = rollout(env, s0, policy)

        total_moves = len(traj) - 1
        p1_moves = sum((s[0], s[1]) != (sn[0], sn[1]) for s, sn in zip(traj[:-1], traj[1:]))
        p2_moves = sum((s[2], s[3]) != (sn[2], sn[3]) for s, sn in zip(traj[:-1], traj[1:]))
        unique_states = len(set(traj))

        draw_trajectory(ax, traj, grid_size)

        fig.suptitle(f"Planning DQN Rollout from {s0}", fontsize=14, y=0.97)
        ax.set_title(
            f"Total moves: {total_moves} | "
            f"P1 moves: {p1_moves} | "
            f"P2 moves: {p2_moves} | "
            f"Unique states: {unique_states}",
            fontsize=10,
            color="gray",
            pad=6,
        )

        fig.canvas.draw_idle()

    if not is_headless:
        from matplotlib.widgets import Button

        plt.subplots_adjust(bottom=0.2)

        axprev = plt.axes([0.25, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.65, 0.05, 0.1, 0.075])

        bprev = Button(axprev, "Prev")
        bnext = Button(axnext, "Next")

        def prev_state(_event):
            idx[0] = (idx[0] - 1) % len(valid_states)
            update_plot()

        def next_state(_event):
            idx[0] = (idx[0] + 1) % len(valid_states)
            update_plot()

        bprev.on_clicked(prev_state)
        bnext.on_clicked(next_state)

        update_plot()
        plt.show()
    else:
        sample_states = random.sample(valid_states, min(5, len(valid_states)))
        print(f"Headless mode detected (backend={plt.get_backend()}). Saving 5 rollouts...")
        for i, s0 in enumerate(sample_states):
            idx[0] = valid_states.index(s0)
            update_plot()
            rollout_file = output_path(f"rollout_{i + 1}.png")
            fig.savefig(rollout_file, dpi=150, bbox_inches="tight")
            print(f"  Saved {rollout_file} from {s0}")


def plot_training_loss(losses, output_path):
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    is_headless = plt.get_backend().lower() == "agg" or not has_display

    plt.figure()
    plt.plot(losses)
    plt.title("Planning Loss")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")

    if not is_headless:
        plt.show()
    else:
        loss_file = output_path("planning_loss.png")
        plt.savefig(loss_file, dpi=150, bbox_inches="tight")
        print(f"Saved {loss_file}")
