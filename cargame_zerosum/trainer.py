import random

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
        TARGET_UPDATE_EVERY,
        device,
    )
    from dqn import DQN, ReplayBuffer, encode_state


def neural_planning(env, iterations=8000):
    """
    Instead of a tabular dictionary, we train the DQN by sampling
    states and using the environment rules to compute targets.
    """
    net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()
    replay_buffer = ReplayBuffer(capacity=10000)

    losses = []

    print("Starting Neural Planning (Function Approximation)...")
    for i in range(iterations):
        # 1. Sample any random state from the environment
        s = random.choice(env.states)
        if (s[0], s[1]) == (s[2], s[3]):
            continue

        s_tensor = encode_state(s, env.grid_size)

        # 2. Compute target for all 16 joint actions using the model (env)
        with torch.no_grad():
            next_states = []
            rewards = []
            terminal_mask = []

            for a_idx in range(16):
                a1, a2 = a_idx // 4, a_idx % 4
                s_next = env.transition(s, a1, a2)
                r = env.reward(s, a1, a2)
                next_states.append(s_next)
                rewards.append(r)
                terminal_mask.append((s_next[0], s_next[1]) == (s_next[2], s_next[3]))

            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            terminal_mask = torch.tensor(terminal_mask, dtype=torch.bool, device=device)

            next_tensors = torch.stack([encode_state(ns, env.grid_size) for ns in next_states])
            q_next_all = target_net(next_tensors).view(16, 4, 4)

            # Minimax: max over a1 of min over a2
            v_next_all = torch.max(torch.min(q_next_all, dim=2)[0], dim=1)[0]
            v_next_all[terminal_mask] = 0.0
            target_q_all = rewards + GAMMA * v_next_all

        replay_buffer.push(s_tensor, target_q_all)

        # 3. Update network (sample mini-batch from buffer)
        if len(replay_buffer) >= MIN_BUFFER_SIZE:
            for _ in range(GRADIENT_STEPS):
                batch_states, batch_targets = replay_buffer.sample(BATCH_SIZE)
                q_pred = net(batch_states)
                loss = loss_fn(q_pred, batch_targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            losses.append(loss.item())

            if (i + 1) % TARGET_UPDATE_EVERY == 0:
                target_net.load_state_dict(net.state_dict())

        if i % 1000 == 0 and losses:
            print(f"Planning Step {i} | Loss: {losses[-1]:.6f}")

    return net, losses
