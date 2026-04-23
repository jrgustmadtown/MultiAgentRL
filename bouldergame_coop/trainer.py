import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .config import (
        BATCH_SIZE,
        GAMMA,
        GRAD_CLIP_NORM,
        GRADIENT_STEPS,
        LR,
        MIN_BUFFER_SIZE,
        NUM_ACTIONS,
        NUM_JOINT_ACTIONS,
        TARGET_UPDATE_EVERY,
        device,
    )
    from .dqn import DQN, ReplayBuffer, encode_state
except ImportError:
    from config import (
        BATCH_SIZE,
        GAMMA,
        GRAD_CLIP_NORM,
        GRADIENT_STEPS,
        LR,
        MIN_BUFFER_SIZE,
        NUM_ACTIONS,
        NUM_JOINT_ACTIONS,
        TARGET_UPDATE_EVERY,
        device,
    )
    from dqn import DQN, ReplayBuffer, encode_state


def neural_planning(env, iterations=8000):
    """Train two joint-action Q-networks with role-weighted cooperative rewards."""
    net1 = DQN().to(device)
    net2 = DQN().to(device)
    target_net1 = DQN().to(device)
    target_net2 = DQN().to(device)
    target_net1.load_state_dict(net1.state_dict())
    target_net2.load_state_dict(net2.state_dict())

    optimizer1 = optim.Adam(net1.parameters(), lr=LR)
    optimizer2 = optim.Adam(net2.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()
    replay_buffer1 = ReplayBuffer(capacity=10000)
    replay_buffer2 = ReplayBuffer(capacity=10000)

    losses1 = []
    losses2 = []

    print("Starting Neural Planning (Boulder Cooperative)...")
    for i in range(iterations):
        s = env.sample_state()
        s_tensor = encode_state(s)

        with torch.no_grad():
            next_states = []
            rewards1 = []
            rewards2 = []
            terminal_mask = []

            for a_idx in range(NUM_JOINT_ACTIONS):
                a1 = a_idx // NUM_ACTIONS
                a2 = a_idx % NUM_ACTIONS
                s_next = env.transition(s, a1, a2)
                r1, r2 = env.reward(s, a1, a2)
                next_states.append(s_next)
                rewards1.append(r1)
                rewards2.append(r2)
                terminal_mask.append(env.is_success(s_next))

            rewards1 = torch.tensor(rewards1, dtype=torch.float32, device=device)
            rewards2 = torch.tensor(rewards2, dtype=torch.float32, device=device)
            terminal_mask = torch.tensor(terminal_mask, dtype=torch.bool, device=device)

            next_tensors = torch.stack([encode_state(ns) for ns in next_states])

            q1_next_all = target_net1(next_tensors)
            q2_next_all = target_net2(next_tensors)

            v1_next = torch.max(q1_next_all, dim=1).values
            v2_next = torch.max(q2_next_all, dim=1).values
            v1_next[terminal_mask] = 0.0
            v2_next[terminal_mask] = 0.0

            target_q1 = rewards1 + GAMMA * v1_next
            target_q2 = rewards2 + GAMMA * v2_next

        replay_buffer1.push(s_tensor, target_q1)
        replay_buffer2.push(s_tensor, target_q2)

        if len(replay_buffer1) >= MIN_BUFFER_SIZE:
            for _ in range(GRADIENT_STEPS):
                batch_states, batch_targets = replay_buffer1.sample(BATCH_SIZE)
                q_pred = net1(batch_states)
                loss1 = loss_fn(q_pred, batch_targets)
                optimizer1.zero_grad()
                loss1.backward()
                torch.nn.utils.clip_grad_norm_(net1.parameters(), GRAD_CLIP_NORM)
                optimizer1.step()

                batch_states, batch_targets = replay_buffer2.sample(BATCH_SIZE)
                q_pred = net2(batch_states)
                loss2 = loss_fn(q_pred, batch_targets)
                optimizer2.zero_grad()
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(net2.parameters(), GRAD_CLIP_NORM)
                optimizer2.step()

            losses1.append(loss1.item())
            losses2.append(loss2.item())

            if (i + 1) % TARGET_UPDATE_EVERY == 0:
                target_net1.load_state_dict(net1.state_dict())
                target_net2.load_state_dict(net2.state_dict())

        if i % 500 == 0 and losses1:
            print(f"Step {i} | P1 Loss: {losses1[-1]:.6f} | P2 Loss: {losses2[-1]:.6f}")

    return (net1, net2), (losses1, losses2)
