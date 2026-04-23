import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from utilities.game_theory import fast_nash_value as _fast_nash_value
from utilities.game_theory import solve_nash as _solve_nash


def solve_nash(q1, q2, num_actions):
    return _solve_nash(q1, q2, num_actions=num_actions)


def fast_nash_value(q1, q2, iters=5):
    return _fast_nash_value(q1, q2, iters=iters)
