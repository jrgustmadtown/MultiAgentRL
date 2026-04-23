"""
Dog Game - Nash DQN

Two-player general-sum game on a continuous 1x1 field.
- Each player has a doghouse (fixed positions)
- Dog position = midpoint of the two players
- Goal: Move so the dog is closer to YOUR house

Continuous space approximated via discrete step size.
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
        DEFAULT_HOUSE1,
        DEFAULT_HOUSE2,
        STEP_SIZE,
        warnings_filter_message,
    )
    from .environment import DogGame
    from .io_utils import export_weights, output_path
    from .policy import get_policy
    from .trainer import neural_planning
    from .visualization import (
        draw_vector_field,
        plot_training_losses,
        run_rollout_visualization,
    )
except ImportError:
    from config import (
        DEFAULT_HORIZON,
        DEFAULT_HOUSE1,
        DEFAULT_HOUSE2,
        STEP_SIZE,
        warnings_filter_message,
    )
    from environment import DogGame
    from io_utils import export_weights, output_path
    from policy import get_policy
    from trainer import neural_planning
    from visualization import draw_vector_field, plot_training_losses, run_rollout_visualization

warnings.filterwarnings("ignore", message=warnings_filter_message)


def parse_position(s):
    """Parse 'x,y' string into (float, float) tuple."""
    x, y = s.split(",")
    return (float(x), float(y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dog Game (Nash-Q)")
    parser.add_argument("--iterations", type=int, default=8000, help="Training iterations")
    parser.add_argument("--step-size", type=float, default=STEP_SIZE, help="Movement step size")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Episode length")
    parser.add_argument(
        "--house1",
        type=str,
        default=f"{DEFAULT_HOUSE1[0]},{DEFAULT_HOUSE1[1]}",
        help="House 1 position (x,y)",
    )
    parser.add_argument(
        "--house2",
        type=str,
        default=f"{DEFAULT_HOUSE2[0]},{DEFAULT_HOUSE2[1]}",
        help="House 2 position (x,y)",
    )
    args = parser.parse_args()

    house1 = parse_position(args.house1)
    house2 = parse_position(args.house2)

    env = DogGame(step_size=args.step_size, house1=house1, house2=house2)

    nets, losses = neural_planning(env, iterations=args.iterations)
    net1, net2 = nets
    losses1, losses2 = losses
    policy_fn = get_policy(nets)

    export_weights(net1, output_path("weights_player1.txt"), "Player 1 Q-network (Dog Game)")
    export_weights(net2, output_path("weights_player2.txt"), "Player 2 Q-network (Dog Game)")

    draw_vector_field(nets, house1, house2)
    plot_training_losses(losses1, losses2)
    run_rollout_visualization(env, policy_fn, house1, house2, args.horizon)
