import numpy as np
import torch

try:
    from .dqn import encode_state
    from .game_theory import solve_nash
except ImportError:
    from dqn import encode_state
    from game_theory import solve_nash


def get_policy(nets, env):
    net1, net2 = nets
    policy = {}
    fallback_count = 0

    for s in env.states:
        s_t = encode_state(s, env.grid_size)
        with torch.no_grad():
            q1 = net1(s_t).view(4, 4)
            q2 = net2(s_t).view(4, 4)

        pi1, pi2, used_fallback = solve_nash(q1, q2)
        if used_fallback:
            fallback_count += 1

        a1 = int(np.argmax(pi1))
        a2 = int(np.argmax(pi2))
        policy[s] = (lambda a=a1: a, lambda a=a2: a)

    print(
        f"Nash fallback used: {fallback_count}/{len(env.states)} states "
        f"({100 * fallback_count / len(env.states):.1f}%)"
    )
    return policy


def rollout(env, s0, policy, horizon=20):
    traj = [s0]
    s = s0
    for _ in range(horizon):
        if (s[0], s[1]) == (s[2], s[3]):
            break
        a1, a2 = policy[s][0](), policy[s][1]()
        s = env.transition(s, a1, a2)
        traj.append(s)
    return traj
