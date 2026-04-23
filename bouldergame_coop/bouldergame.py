"""Boulder Cooperative Game (Scaffold).

Step 2 scaffolding complete.
Step 3+ will implement environment dynamics and training.
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
        DEFAULT_HOLE,
        DEFAULT_ITERATIONS,
        STEP_SIZE,
        warnings_filter_message,
    )
    from .environment import BoulderGame, parse_position
    from .io_utils import export_weights, output_path
    from .policy import get_policy
    from .trainer import neural_planning
    from .visualization import plot_training_losses, run_rollout_visualization
except ImportError:
    from config import (
        DEFAULT_HORIZON,
        DEFAULT_HOLE,
        DEFAULT_ITERATIONS,
        STEP_SIZE,
        warnings_filter_message,
    )
    from environment import BoulderGame, parse_position
    from io_utils import export_weights, output_path
    from policy import get_policy
    from trainer import neural_planning
    from visualization import plot_training_losses, run_rollout_visualization

warnings.filterwarnings("ignore", message=warnings_filter_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boulder Cooperative Game (Scaffold)")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Training iterations")
    parser.add_argument("--step-size", type=float, default=STEP_SIZE, help="Movement step size")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Episode length")
    parser.add_argument(
        "--hole",
        type=str,
        default=f"{DEFAULT_HOLE[0]},{DEFAULT_HOLE[1]}",
        help="Hole position (x,y)",
    )
    parser.add_argument(
        "--random-rollout-starts",
        action="store_true",
        help="Use random initial states for rollout visualization",
    )
    args = parser.parse_args()

    hole = parse_position(args.hole)
    env = BoulderGame(step_size=args.step_size, hole=hole)

    nets, losses = neural_planning(env, iterations=args.iterations)
    net1, net2 = nets
    losses1, losses2 = losses

    export_weights(net1, output_path("weights_player1.txt"), "Player 1 Q-network (x-axis role)")
    export_weights(net2, output_path("weights_player2.txt"), "Player 2 Q-network (y-axis role)")

    policy_fn = get_policy(nets, env)
    plot_training_losses(losses1, losses2, output_path)
    run_rollout_visualization(
        env,
        policy_fn,
        args.horizon,
        output_path,
        use_random_starts=args.random_rollout_starts,
    )

    print(f"Training complete. Loss points: P1={len(losses1)} P2={len(losses2)}")
