import torch

try:
    from .config import DEFAULT_HORIZON, NUM_ACTIONS
    from .dqn import encode_state
except ImportError:
    from config import DEFAULT_HORIZON, NUM_ACTIONS
    from dqn import encode_state


def get_policy(nets, _env=None, _use_exact_nash=False):
    """Return a cooperative policy function from two Q-networks.

    Action is selected by maximizing joint utility q1 + q2.
    """
    net1, net2 = nets

    def policy_fn(s):
        s_t = encode_state(s)
        with torch.no_grad():
            q1 = net1(s_t).view(NUM_ACTIONS, NUM_ACTIONS)
            q2 = net2(s_t).view(NUM_ACTIONS, NUM_ACTIONS)
            joint = q1 + q2
            a_idx = torch.argmax(joint).item()
            a1 = a_idx // NUM_ACTIONS
            a2 = a_idx % NUM_ACTIONS
        return a1, a2

    return policy_fn


def rollout(env, s0, policy_fn, horizon=None):
    """Run a trajectory from s0 using the policy function."""
    if horizon is None:
        horizon = DEFAULT_HORIZON
    traj = [s0]
    s = s0
    for _ in range(horizon):
        if env.is_success(s):
            break
        a1, a2 = policy_fn(s)
        s = env.transition(s, a1, a2)
        traj.append(s)
    return traj
