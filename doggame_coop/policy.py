import torch

try:
    from .config import NUM_ACTIONS
    from .dqn import encode_state
except ImportError:
    from config import NUM_ACTIONS
    from dqn import encode_state


def get_policy(nets):
    """Cooperative policy selecting joint action that maximizes q1+q2."""
    net1, net2 = nets

    def policy_fn(s):
        s_t = encode_state(s)
        with torch.no_grad():
            q1 = net1(s_t)
            q2 = net2(s_t)
            joint_sum = q1 + q2
            best_idx = int(torch.argmax(joint_sum).item())

        a1 = best_idx // NUM_ACTIONS
        a2 = best_idx % NUM_ACTIONS
        return a1, a2

    return policy_fn


def rollout(env, s0, policy_fn, horizon):
    """Run one trajectory, stopping early if the dog reaches the house."""
    traj = [s0]
    s = s0
    for _ in range(horizon):
        a1, a2 = policy_fn(s)
        s = env.transition(s, a1, a2)
        traj.append(s)
        if env.is_success(s):
            break
    return traj
