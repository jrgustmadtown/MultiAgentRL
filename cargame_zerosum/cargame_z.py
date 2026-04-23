"""
Zero-Sum Car Game - Minimax DQN

Two-player zero-sum game where Player 1 (maximizer) and Player 2 (minimizer)
navigate on a grid. Uses minimax Q-learning: max_a1 min_a2 Q(s,a1,a2)
"""
import argparse
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

try:
    from .config import GRID_SIZE
    from .environment import CarGame
    from .io_utils import export_weights, output_path
    from .policy import get_policy
    from .trainer import neural_planning
    from .visualization import plot_training_loss, run_rollout_visualization
except ImportError:
    from config import GRID_SIZE
    from environment import CarGame
    from io_utils import export_weights, output_path
    from policy import get_policy
    from trainer import neural_planning
    from visualization import plot_training_loss, run_rollout_visualization


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Sum Car Game (Minimax DQN)")
    parser.add_argument("--iterations", type=int, default=8000, help="Number of planning iterations")
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE, help="Size of the grid (default: 5)")
    args = parser.parse_args()

    env = CarGame(grid_size=args.grid_size)

    net, losses = neural_planning(env, iterations=args.iterations)
    policy = get_policy(net, env)

    export_weights(
        net,
        output_path("weights_player1.txt"),
        "Player 1 (maximizer): argmax over a1 of min over a2",
    )
    export_weights(
        net,
        output_path("weights_player2.txt"),
        "Player 2 (minimizer): opponent uses argmin over a2 given P1's a1",
    )

    run_rollout_visualization(env, policy, args.grid_size, output_path)
    plot_training_loss(losses, output_path)
