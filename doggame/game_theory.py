import numpy as np
import nashpy as nash


def solve_nash(q1, q2, num_actions):
    """Returns (pi1, pi2, used_fallback)."""
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

    uniform = np.ones(num_actions) / num_actions
    return uniform, uniform, True


def fast_nash_value(q1, q2, iters=5):
    """Fast approximation of Nash value using iterated best response."""
    a1 = q1.max(dim=1).values.argmax().item()
    a2 = q2.max(dim=0).values.argmax().item()
    for _ in range(iters):
        a1_new = q1[:, a2].argmax().item()
        a2_new = q2[a1, :].argmax().item()
        if a1_new == a1 and a2_new == a2:
            break
        a1, a2 = a1_new, a2_new
    return q1[a1, a2], q2[a1, a2]
