"""
Dog Game Coop - Planning DQN

Two-player cooperative game on a continuous 1x1 field.
- Dog position = midpoint of the two players
- Single dog house target in the upper-right by default
- Both players share the same reward objective
"""

import argparse
import os
import sys
import warnings


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

try:
    from .config import (
        DEFAULT_HORIZON,
        DEFAULT_HOUSE,
        DEFAULT_ITERATIONS,
        DEFAULT_WALL,
        STEP_SIZE,
        SUCCESS_RADIUS,
        warnings_filter_message,
    )
    from .environment import DogGameCoop
    from .io_utils import export_weights, output_path
    from .policy import get_policy
    from .trainer import neural_planning
    from .visualization import plot_training_losses, run_rollout_visualization
except ImportError:
    from config import (
        DEFAULT_HORIZON,
        DEFAULT_HOUSE,
        DEFAULT_ITERATIONS,
        DEFAULT_WALL,
        STEP_SIZE,
        SUCCESS_RADIUS,
        warnings_filter_message,
    )
    from environment import DogGameCoop
    from io_utils import export_weights, output_path
    from policy import get_policy
    from trainer import neural_planning
    from visualization import plot_training_losses, run_rollout_visualization

warnings.filterwarnings("ignore", message=warnings_filter_message)


def parse_position(s):
    x, y = s.split(",")
    return (float(x), float(y))


def parse_wall(s):
    x0, y0, x1, y1 = s.split(",")
    return ((float(x0), float(y0)), (float(x1), float(y1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dog Game Coop")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Training iterations")
    parser.add_argument("--step-size", type=float, default=STEP_SIZE, help="Movement step size")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Rollout horizon")
    parser.add_argument(
        "--house",
        type=str,
        default=f"{DEFAULT_HOUSE[0]},{DEFAULT_HOUSE[1]}",
        help="Single house position (x,y)",
    )
    parser.add_argument(
        "--success-radius",
        type=float,
        default=SUCCESS_RADIUS,
        help="Terminal radius around the house",
    )
    parser.add_argument(
        "--wall",
        type=str,
        default=f"{DEFAULT_WALL[0][0]},{DEFAULT_WALL[0][1]},{DEFAULT_WALL[1][0]},{DEFAULT_WALL[1][1]}",
        help="Wall segment as x0,y0,x1,y1",
    )
    args = parser.parse_args()

    house = parse_position(args.house)
    wall = parse_wall(args.wall)

    env = DogGameCoop(
        step_size=args.step_size,
        house=house,
        wall=wall,
        success_radius=args.success_radius,
    )

    nets, losses = neural_planning(env, iterations=args.iterations)
    net1, net2 = nets
    losses1, losses2 = losses
    policy_fn = get_policy(nets)

    export_weights(net1, output_path("weights_player1.txt"), "Player 1 Q-network (Dog Game Coop)")
    export_weights(net2, output_path("weights_player2.txt"), "Player 2 Q-network (Dog Game Coop)")

    plot_training_losses(losses1, losses2)
    run_rollout_visualization(env, policy_fn, house, args.horizon)
