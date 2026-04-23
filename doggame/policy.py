import numpy as np
import torch

try:
    from .config import NUM_ACTIONS
    from .dqn import encode_state
    from .game_theory import solve_nash
except ImportError:
    from config import NUM_ACTIONS
    from dqn import encode_state
    from game_theory import solve_nash


def get_policy(nets, use_exact_nash=False):
    """Return a function that computes policy actions for any continuous state."""
    net1, net2 = nets

    def policy_fn(s):
        s_t = encode_state(s)
        with torch.no_grad():
            q1 = net1(s_t).view(NUM_ACTIONS, NUM_ACTIONS)
            q2 = net2(s_t).view(NUM_ACTIONS, NUM_ACTIONS)

        if use_exact_nash:
            pi1, pi2, _ = solve_nash(q1, q2, NUM_ACTIONS)
            a1 = int(np.argmax(pi1))
            a2 = int(np.argmax(pi2))
        else:
            a1 = q1.max(dim=1).values.argmax().item()
            a2 = q2.max(dim=0).values.argmax().item()
            for _ in range(5):
                a1_new = q1[:, a2].argmax().item()
                a2_new = q2[a1, :].argmax().item()
                if a1_new == a1 and a2_new == a2:
                    break
                a1, a2 = a1_new, a2_new

        return a1, a2

    return policy_fn


def rollout(env, s0, policy_fn, horizon):
    """Run a trajectory from s0 using the policy function."""
    traj = [s0]
    s = s0
    for _ in range(horizon):
        a1, a2 = policy_fn(s)
        s = env.transition(s, a1, a2)
        traj.append(s)
    return traj
