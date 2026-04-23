import numpy as np
import nashpy as nash


def solve_nash(q1, q2):
    """
    Compute Nash equilibrium for a 4x4 bimatrix game.
    q1[a1, a2] = P1's payoff, q2[a1, a2] = P2's payoff.
    Returns (pi1, pi2, used_fallback).
    """
    q1_np = q1.cpu().numpy() if hasattr(q1, "cpu") else q1
    q2_np = q2.cpu().numpy() if hasattr(q2, "cpu") else q2

    game = nash.Game(q1_np, q2_np)

    try:
        equilibria = list(game.support_enumeration())
        if equilibria:
            pi1, pi2 = equilibria[0]
            return pi1, pi2, False
    except Exception:
        pass

    uniform = np.ones(4) / 4
    return uniform, uniform, True


def fast_nash_value(q1, q2):
    """
    Fast approximation of Nash value using iterated best response.
    Returns (V1, V2) values at the approximate equilibrium.
    """
    a1, a2 = 0, 0
    for _ in range(3):
        a1_new = q1[:, a2].argmax().item()
        a2_new = q2[a1, :].argmax().item()
        if a1_new == a1 and a2_new == a2:
            break
        a1, a2 = a1_new, a2_new
    return q1[a1, a2], q2[a1, a2]
