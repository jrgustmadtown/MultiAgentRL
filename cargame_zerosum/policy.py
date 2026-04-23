import torch

try:
    from .dqn import encode_state
except ImportError:
    from dqn import encode_state


def get_policy(net, env):
    policy = {}
    for s in env.states:
        s_t = encode_state(s, env.grid_size)
        with torch.no_grad():
            q = net(s_t).view(4, 4)
        a1 = torch.argmax(torch.min(q, dim=1)[0]).item()
        a2 = torch.argmin(q[a1]).item()
        policy[s] = (lambda a=a1: a, lambda a=a2: a)
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
