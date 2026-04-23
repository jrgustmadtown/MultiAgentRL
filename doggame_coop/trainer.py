import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .config import (
        BATCH_SIZE,
        DEFAULT_ITERATIONS,
        GAMMA,
        GRAD_CLIP_NORM,
        GRADIENT_STEPS,
        LR,
        MIN_BUFFER_SIZE,
        NUM_ACTIONS,
        NUM_JOINT_ACTIONS,
        TAU,
        device,
    )
    from .dqn import DQN, ReplayBuffer, encode_state
except ImportError:
    from config import (
        BATCH_SIZE,
        DEFAULT_ITERATIONS,
        GAMMA,
        GRAD_CLIP_NORM,
        GRADIENT_STEPS,
        LR,
        MIN_BUFFER_SIZE,
        NUM_ACTIONS,
        NUM_JOINT_ACTIONS,
        TAU,
        device,
    )
    from dqn import DQN, ReplayBuffer, encode_state


def neural_planning(env, iterations=DEFAULT_ITERATIONS):
    """Train two cooperative Q-networks via planning over sampled states."""
    net1 = DQN().to(device)
    net2 = DQN().to(device)
    target_net1 = DQN().to(device)
    target_net2 = DQN().to(device)
    target_net1.load_state_dict(net1.state_dict())
    target_net2.load_state_dict(net2.state_dict())

    optimizer1 = optim.Adam(net1.parameters(), lr=LR)
    optimizer2 = optim.Adam(net2.parameters(), lr=LR)

    warmup_steps = 200

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.9999 ** (step - warmup_steps)

    scheduler1 = optim.lr_scheduler.LambdaLR(optimizer1, lr_lambda)
    scheduler2 = optim.lr_scheduler.LambdaLR(optimizer2, lr_lambda)

    loss_fn = nn.SmoothL1Loss()
    replay_buffer1 = ReplayBuffer(capacity=10000)
    replay_buffer2 = ReplayBuffer(capacity=10000)

    losses1 = []
    losses2 = []

    print("Starting Neural Planning (Cooperative Dog Game)...")
    for i in range(iterations):
        s = env.sample_state()
        s_tensor = encode_state(s)

        with torch.no_grad():
            next_states = []
            rewards = []

            for a_idx in range(NUM_JOINT_ACTIONS):
                a1 = a_idx // NUM_ACTIONS
                a2 = a_idx % NUM_ACTIONS
                s_next = env.transition(s, a1, a2)
                r, _ = env.reward(s, a1, a2)
                next_states.append(s_next)
                rewards.append(r)

            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            next_tensors = torch.stack([encode_state(ns) for ns in next_states])

            q1_next_all = target_net1(next_tensors)
            q2_next_all = target_net2(next_tensors)
            coop_sum = q1_next_all + q2_next_all
            best_idx = torch.argmax(coop_sum, dim=1, keepdim=True)

            v1_next = q1_next_all.gather(1, best_idx).squeeze(1)
            v2_next = q2_next_all.gather(1, best_idx).squeeze(1)

            target_q1 = rewards + GAMMA * v1_next
            target_q2 = rewards + GAMMA * v2_next

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

            for target_param, param in zip(target_net1.parameters(), net1.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            for target_param, param in zip(target_net2.parameters(), net2.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            scheduler1.step()
            scheduler2.step()

        if i % 1000 == 0 and losses1:
            print(
                f"Step {i} | P1 Loss: {losses1[-1]:.6f} | "
                f"P2 Loss: {losses2[-1]:.6f} | LR: {scheduler1.get_last_lr()[0]:.6f}"
            )

    return (net1, net2), (losses1, losses2)
