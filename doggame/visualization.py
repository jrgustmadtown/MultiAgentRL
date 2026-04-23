import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from utilities.runtime import is_headless_matplotlib

try:
    from .config import ACTION_DIRS, NUM_ACTIONS
    from .dqn import encode_state
    from .environment import distance, dog_position
    from .io_utils import output_path
    from .policy import rollout
except ImportError:
    from config import ACTION_DIRS, NUM_ACTIONS
    from dqn import encode_state
    from environment import distance, dog_position
    from io_utils import output_path
    from policy import rollout


def draw_trajectory(ax, traj, house1, house2):
    """Draw a trajectory on a 1x1 field."""
    ax.clear()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")

    ax.plot(*house1, "s", color="red", markersize=15, alpha=0.7)
    ax.plot(*house2, "s", color="blue", markersize=15, alpha=0.7)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "k-", linewidth=2)

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


def draw_vector_field(nets, house1, house2, grid_res=15):
    """Draw vector fields showing each player's policy direction."""
    net1, net2 = nets

    xs = np.linspace(0.05, 0.95, grid_res)
    ys = np.linspace(0.05, 0.95, grid_res)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for player_idx, ax in enumerate(axes):
        u = np.zeros((grid_res, grid_res))
        v = np.zeros((grid_res, grid_res))

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                if player_idx == 0:
                    s = (x, y, 1.0, 1.0)
                else:
                    s = (0.0, 0.0, x, y)

                s_t = encode_state(s)
                with torch.no_grad():
                    q1 = net1(s_t).view(NUM_ACTIONS, NUM_ACTIONS)
                    q2 = net2(s_t).view(NUM_ACTIONS, NUM_ACTIONS)

                a1 = q1.max(dim=1).values.argmax().item()
                a2 = q2.max(dim=0).values.argmax().item()
                for _ in range(5):
                    a1_new = q1[:, a2].argmax().item()
                    a2_new = q2[a1, :].argmax().item()
                    if a1_new == a1 and a2_new == a2:
                        break
                    a1, a2 = a1_new, a2_new

                action = a1 if player_idx == 0 else a2
                dx, dy = ACTION_DIRS[action]
                u[j, i] = dx
                v[j, i] = dy

        x_mesh, y_mesh = np.meshgrid(xs, ys)

        color = "red" if player_idx == 0 else "blue"
        ax.quiver(x_mesh, y_mesh, u, v, color=color, alpha=0.7, scale=20)

        ax.plot(*house1, "s", color="red", markersize=12, alpha=0.7)
        ax.plot(*house2, "s", color="blue", markersize=12, alpha=0.7)
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "k-", linewidth=2)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")

        opponent_pos = "(1,1)" if player_idx == 0 else "(0,0)"
        ax.set_title(f"Player {player_idx + 1} Policy (opponent at {opponent_pos})")

    plt.tight_layout()
    vector_path = output_path("vector_fields.png")
    plt.savefig(vector_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {vector_path}")


def plot_training_losses(losses1, losses2):
    plt.figure()
    plt.plot(losses1, label="P1 Loss", alpha=0.7)
    plt.plot(losses2, label="P2 Loss", alpha=0.7)
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Dog Game - Nash-Q Training Loss")
    plt.legend()
    loss_path = output_path("planning_loss.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"Saved {loss_path}")


def run_rollout_visualization(env, policy_fn, house1, house2, horizon):
    is_headless = is_headless_matplotlib(plt)

    fig, ax = plt.subplots(figsize=(8, 8))

    if is_headless:
        print("Headless mode. Saving 5 rollouts...")
        for i in range(5):
            s0 = env.sample_state()
            traj = rollout(env, s0, policy_fn, horizon)
            draw_trajectory(ax, traj, house1, house2)

            final_dog = dog_position(traj[-1])
            d1 = distance(final_dog, house1)
            d2 = distance(final_dog, house2)

            ax.set_title(f"Rollout {i + 1}: Dog to H1={d1:.3f}, to H2={d2:.3f}")
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
        draw_trajectory(ax, traj, house1, house2)

        final_dog = dog_position(traj[-1])
        d1 = distance(final_dog, house1)
        d2 = distance(final_dog, house2)

        fig.suptitle("Dog Game Rollout", fontsize=14)
        ax.set_title(f"Dog to House1: {d1:.3f} | Dog to House2: {d2:.3f}", fontsize=10)
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
