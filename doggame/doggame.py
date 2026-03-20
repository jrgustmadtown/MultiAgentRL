"""
Dog Game - Nash DQN

Two-player general-sum game on a continuous 1x1 field.
- Each player has a doghouse (fixed positions)
- Dog position = midpoint of the two players
- Goal: Move so the dog is closer to YOUR house

Continuous space approximated via discrete step size.
"""
import argparse
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nashpy as nash
import random
import os
from collections import deque
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*equilibria was returned.*")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LR = 5e-4
STEP_SIZE = 0.1  # Movement step in [0,1] space
GAMMA = 0.5
TAU = 0.001  # Polyak averaging coefficient for soft target updates
GRAD_CLIP_NORM = 1.0
BATCH_SIZE = 32
MIN_BUFFER_SIZE = 64
GRADIENT_STEPS = 4
HORIZON = 10  # Fixed episode length

# House positions
HOUSE1 = (0.25, 0.25)
HOUSE2 = (0.75, 0.75)

# 17 actions, stay + 16 directions - every 22.5 degrees
# unit vectors
# 0° = East - counterclockwise
_angles_deg = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 
               180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]
ACTION_DIRS = {0: (0, 0)} 
for i, deg in enumerate(_angles_deg):
    rad = np.radians(deg)
    ACTION_DIRS[i + 1] = (np.cos(rad), np.sin(rad))

ACTION_NAMES = ['Stay', 'E', 'ENE', 'NE', 'NNE', 'N', 'NNW', 'NW', 'WNW',
                'W', 'WSW', 'SW', 'SSW', 'S', 'SSE', 'SE', 'ESE']
NUM_ACTIONS = 17


def encode_state(s):
    """
    State s = (p1_x, p1_y, p2_x, p2_y) in [0,1] space.
    Already normalized for NN input.
    """
    return torch.tensor(s, dtype=torch.float32, device=device)


def dog_position(s):
    """Dog is at the midpoint of the two players."""
    return ((s[0] + s[2]) / 2, (s[1] + s[3]) / 2)


def distance(p1, p2):
    """Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def solve_nash(Q1, Q2, track=False):
    """
    Returns (pi1, pi2) mixed strategies as numpy arrays.
    falls back to uniform if no equilibrium found.
    """
    Q1_np = Q1.cpu().numpy() if isinstance(Q1, torch.Tensor) else Q1
    Q2_np = Q2.cpu().numpy() if isinstance(Q2, torch.Tensor) else Q2
    
    game = nash.Game(Q1_np, Q2_np)
    
    try:
        equilibria = list(game.support_enumeration())
        if equilibria:
            pi1, pi2 = equilibria[0]
            return pi1, pi2, False
    except:
        pass
    
    uniform = np.ones(NUM_ACTIONS) / NUM_ACTIONS
    return uniform, uniform, True


def fast_nash_value(Q1, Q2):
    """
    Fast approximation of Nash value using iterated best response.
    Returns (V1, V2) values at the approximate equilibrium.
    """
    # Start from greedy individual bests
    a1 = Q1.max(dim=1).values.argmax().item()
    a2 = Q2.max(dim=0).values.argmax().item()
    for _ in range(5):
        a1_new = Q1[:, a2].argmax().item()
        a2_new = Q2[a1, :].argmax().item()
        if a1_new == a1 and a2_new == a2:
            break
        a1, a2 = a1_new, a2_new
    return Q1[a1, a2], Q2[a1, a2]

"""Neural Network that approximates Q(s, a1, a2)."""
class DQN(nn.Module):
    def __init__(self, state_dim=4, action_dim=289):  # 17x17 = 289 joint actions
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, s):
        return self.net(s)


class ReplayBuffer:
    """Fixed-size buffer to store (state_tensor, target_q) tuples."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_tensor, target_q):
        self.buffer.append((state_tensor, target_q))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = torch.stack([x[0] for x in batch])
        targets = torch.stack([x[1] for x in batch])
        return states, targets

    def __len__(self):
        return len(self.buffer)


def export_weights(net, filepath, player_info=""):
    """Export network weights with bias as last entry in each row.
    
    Format: Each row is one output neuron with [input_weights..., bias].
    Layers separated by '-----'.
    """
    with open(filepath, 'w') as f:
        layer_idx = 0
        
        for name, module in net.net.named_children():
            if isinstance(module, nn.Linear):
                # W is (output_dim, input_dim), b is (output_dim,)
                W = module.weight.data.cpu().numpy()
                b = module.bias.data.cpu().numpy()
                
                if layer_idx > 0:
                    f.write("-----\n")
                
                # Each row: one output neuron with weights + bias
                for j in range(W.shape[0]):
                    row = list(W[j]) + [b[j]]
                    f.write(",".join(f"{v:.6f}" for v in row) + "\n")
                
                layer_idx += 1
        
        # All comments at end
        f.write("=====\n")
        f.write("# Layer 0: Linear(4 -> 256) - 256 rows x 5 cols (4 weights + bias)\n")
        f.write("# Layer 1: Linear(256 -> 256) - 256 rows x 257 cols (256 weights + bias)\n")
        f.write("# Layer 2: Linear(256 -> 289) - 289 rows x 257 cols (256 weights + bias)\n")
        f.write("# Format: each row is [input_weights..., bias] for one output neuron\n")
        f.write("# Architecture: 4 -> 256 -> 256 -> 289\n")
        f.write("# Activation: ReLU\n")
        f.write("# Actions: 0=Stay, 1=E, 2=ENE, 3=NE, 4=NNE, 5=N, 6=NNW, 7=NW, 8=WNW, 9=W, 10=WSW, 11=SW, 12=SSW, 13=S, 14=SSE, 15=SE, 16=ESE\n")
    
    print(f"Saved weights to {filepath}")


class DogGame:
    """
    Dog Game Environment.
    
    State: (p1_x, p1_y, p2_x, p2_y) - all in [0, 1]
    Actions: 0-8 (stay + 8 directions)
    Dog: midpoint of players
    Reward: -distance(dog, my_house)
    """
    def __init__(self, step_size=0.1, house1=None, house2=None):
        self.step_size = step_size
        self.house1 = house1 if house1 else HOUSE1
        self.house2 = house2 if house2 else HOUSE2
    
    def move(self, x, y, action):
        """Apply action and clamp to [0, 1]."""
        dx, dy = ACTION_DIRS[action]
        x_new = np.clip(x + dx * self.step_size, 0, 1)
        y_new = np.clip(y + dy * self.step_size, 0, 1)
        return x_new, y_new
    
    def transition(self, s, a1, a2):
        """Return next state after both players move."""
        x1_new, y1_new = self.move(s[0], s[1], a1)
        x2_new, y2_new = self.move(s[2], s[3], a2)
        return (x1_new, y1_new, x2_new, y2_new)
    
    def reward(self, s, a1, a2):
        """
        Compute rewards for both players.
        r1 = -distance(dog, house1)
        r2 = -distance(dog, house2)
        """
        s_next = self.transition(s, a1, a2)
        dog = dog_position(s_next)
        r1 = -distance(dog, self.house1)
        r2 = -distance(dog, self.house2)
        return r1, r2
    
    def sample_state(self):
        """
        Sample a random state with bias toward corners/boundaries.
        50% uniform random, 50% near boundaries.
        """
        if random.random() < 0.5:
            # Uniform random
            return (random.random(), random.random(), 
                    random.random(), random.random())
        else:
            # Bias toward corners: use beta distribution that favors 0 and 1
            def biased_coord():
                if random.random() < 0.5:
                    return random.betavariate(0.3, 2)  # Biased toward 0
                else:
                    return random.betavariate(2, 0.3)  # Biased toward 1
            return (biased_coord(), biased_coord(),
                    biased_coord(), biased_coord())


def neural_planning(env, iterations=8000):
    """Train Nash-Q networks by sampling random states."""
    net1 = DQN().to(device)
    net2 = DQN().to(device)
    target_net1 = DQN().to(device)
    target_net2 = DQN().to(device)
    target_net1.load_state_dict(net1.state_dict())
    target_net2.load_state_dict(net2.state_dict())
    optimizer1 = optim.Adam(net1.parameters(), lr=LR)
    optimizer2 = optim.Adam(net2.parameters(), lr=LR)
    # Warmup + decay: linear warmup for 200 steps, then exponential decay
    warmup_steps = 200
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps  # Linear warmup from 0 to 1
        else:
            return 0.9999 ** (step - warmup_steps)  # Exponential decay after
    scheduler1 = optim.lr_scheduler.LambdaLR(optimizer1, lr_lambda)
    scheduler2 = optim.lr_scheduler.LambdaLR(optimizer2, lr_lambda)
    loss_fn = nn.SmoothL1Loss()
    replay_buffer1 = ReplayBuffer(capacity=10000)
    replay_buffer2 = ReplayBuffer(capacity=10000)

    losses1 = []
    losses2 = []
    
    num_joint_actions = NUM_ACTIONS * NUM_ACTIONS  # 289

    print("Starting Neural Planning (Nash-Q for Dog Game)...")
    for i in range(iterations):
        # Sample random state
        s = env.sample_state()
        s_tensor = encode_state(s)
        
        # Compute targets for all 81 joint actions
        with torch.no_grad():
            next_states = []
            rewards1 = []
            rewards2 = []
            
            for a_idx in range(num_joint_actions):
                a1, a2 = a_idx // NUM_ACTIONS, a_idx % NUM_ACTIONS
                s_next = env.transition(s, a1, a2)
                r1, r2 = env.reward(s, a1, a2)
                next_states.append(s_next)
                rewards1.append(r1)
                rewards2.append(r2)
            
            rewards1 = torch.tensor(rewards1, dtype=torch.float32, device=device)
            rewards2 = torch.tensor(rewards2, dtype=torch.float32, device=device)
            
            # Encode all next states
            next_tensors = torch.stack([encode_state(ns) for ns in next_states])
            
            # Forward pass on target networks
            q1_next_all = target_net1(next_tensors).view(num_joint_actions, NUM_ACTIONS, NUM_ACTIONS)
            q2_next_all = target_net2(next_tensors).view(num_joint_actions, NUM_ACTIONS, NUM_ACTIONS)
            
            # Fast Nash approximation
            v1_next = torch.zeros(num_joint_actions, device=device)
            v2_next = torch.zeros(num_joint_actions, device=device)
            for idx in range(num_joint_actions):
                v1_next[idx], v2_next[idx] = fast_nash_value(q1_next_all[idx], q2_next_all[idx])
            
            target_q1 = rewards1 + GAMMA * v1_next
            target_q2 = rewards2 + GAMMA * v2_next

        # Store in replay buffers
        replay_buffer1.push(s_tensor, target_q1)
        replay_buffer2.push(s_tensor, target_q2)

        # Update networks
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

            # Soft target update (Polyak averaging)
            for target_param, param in zip(target_net1.parameters(), net1.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            for target_param, param in zip(target_net2.parameters(), net2.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
            # Gentle LR decay
            scheduler1.step()
            scheduler2.step()

        if i % 1000 == 0 and losses1:
            print(f"Step {i} | P1 Loss: {losses1[-1]:.6f} | P2 Loss: {losses2[-1]:.6f} | LR: {scheduler1.get_last_lr()[0]:.6f}")

    return (net1, net2), (losses1, losses2)


def get_policy(nets, env, use_exact_nash=False):
    """
    Since state space is continuous, we can't enumerate all states.
    Instead, return a function that computes policy for any state.
    
    use_exact_nash: If True, use slow support_enumeration. If False, use fast iterated best response.
    """
    net1, net2 = nets
    
    def policy_fn(s):
        s_t = encode_state(s)
        with torch.no_grad():
            q1 = net1(s_t).view(NUM_ACTIONS, NUM_ACTIONS)
            q2 = net2(s_t).view(NUM_ACTIONS, NUM_ACTIONS)
        
        if use_exact_nash:
            pi1, pi2, _ = solve_nash(q1, q2)
            a1 = int(np.argmax(pi1))
            a2 = int(np.argmax(pi2))
        else:
            # Fast: iterated best response
            # Start from each player's greedy action (max over opponent's actions)
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


def rollout(env, s0, policy_fn, T=None):
    """Run a trajectory from s0 using the policy function."""
    if T is None:
        T = HORIZON
    traj = [s0]
    s = s0
    for _ in range(T):
        a1, a2 = policy_fn(s)
        s = env.transition(s, a1, a2)
        traj.append(s)
    return traj


def draw_trajectory(ax, traj, title=""):
    """Draw the trajectory on a 1x1 field."""
    ax.clear()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    
    # Draw houses
    ax.plot(*HOUSE1, 's', color='red', markersize=15, alpha=0.7)
    ax.plot(*HOUSE2, 's', color='blue', markersize=15, alpha=0.7)
    
    # Draw field boundary
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', linewidth=2)
    
    total_steps = len(traj) - 1
    
    for i, (s, s_next) in enumerate(zip(traj[:-1], traj[1:])):
        alpha = 0.3 + 0.7 * (i / max(1, total_steps))
        
        # Player 1 (red)
        ax.arrow(s[0], s[1], s_next[0] - s[0], s_next[1] - s[1],
                 color='red', alpha=alpha, head_width=0.02, 
                 length_includes_head=True, linewidth=1.5)
        
        # Player 2 (blue)
        ax.arrow(s[2], s[3], s_next[2] - s[2], s_next[3] - s[3],
                 color='blue', alpha=alpha, head_width=0.02,
                 length_includes_head=True, linewidth=1.5)
    
    # Draw dog trajectory (midpoints)
    dog_traj = [dog_position(s) for s in traj]
    for i, (d, d_next) in enumerate(zip(dog_traj[:-1], dog_traj[1:])):
        alpha = 0.3 + 0.7 * (i / max(1, total_steps))
        ax.arrow(d[0], d[1], d_next[0] - d[0], d_next[1] - d[1],
                 color='green', alpha=alpha, head_width=0.015,
                 length_includes_head=True, linewidth=1, linestyle='--')
    
    # Mark final positions
    final = traj[-1]
    ax.plot(final[0], final[1], 'o', color='red', markersize=8)
    ax.plot(final[2], final[3], 'o', color='blue', markersize=8)
    final_dog = dog_position(final)
    ax.plot(final_dog[0], final_dog[1], 'o', color='green', markersize=10)
    
    # Mark start positions
    start = traj[0]
    ax.plot(start[0], start[1], 'x', color='darkred', markersize=8)
    ax.plot(start[2], start[3], 'x', color='darkblue', markersize=8)


def draw_vector_field(nets, env, grid_res=15):
    """
    Draw vector fields showing each player's policy direction.
    For player 1: fix player 2 at (1,1) corner
    For player 2: fix player 1 at (0,0) corner
    """
    net1, net2 = nets
    
    # Create grid
    xs = np.linspace(0.05, 0.95, grid_res)
    ys = np.linspace(0.05, 0.95, grid_res)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for player_idx, ax in enumerate(axes):
        U = np.zeros((grid_res, grid_res))
        V = np.zeros((grid_res, grid_res))
        
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                if player_idx == 0:
                    # Player 1's field: fix P2 at (1, 1)
                    s = (x, y, 1.0, 1.0)
                else:
                    # Player 2's field: fix P1 at (0, 0)
                    s = (0.0, 0.0, x, y)
                
                s_t = encode_state(s)
                with torch.no_grad():
                    q1 = net1(s_t).view(NUM_ACTIONS, NUM_ACTIONS)
                    q2 = net2(s_t).view(NUM_ACTIONS, NUM_ACTIONS)
                
                # Get action via iterated best response
                a1 = q1.max(dim=1).values.argmax().item()
                a2 = q2.max(dim=0).values.argmax().item()
                for _ in range(5):
                    a1_new = q1[:, a2].argmax().item()
                    a2_new = q2[a1, :].argmax().item()
                    if a1_new == a1 and a2_new == a2:
                        break
                    a1, a2 = a1_new, a2_new
                
                # Get direction for this player's action
                action = a1 if player_idx == 0 else a2
                dx, dy = ACTION_DIRS[action]
                U[j, i] = dx
                V[j, i] = dy
        
        X, Y = np.meshgrid(xs, ys)
        
        color = 'red' if player_idx == 0 else 'blue'
        ax.quiver(X, Y, U, V, color=color, alpha=0.7, scale=20)
        
        # Draw houses
        ax.plot(*HOUSE1, 's', color='red', markersize=12, alpha=0.7)
        ax.plot(*HOUSE2, 's', color='blue', markersize=12, alpha=0.7)
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', linewidth=2)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        
        opponent_pos = "(1,1)" if player_idx == 0 else "(0,0)"
        ax.set_title(f"Player {player_idx + 1} Policy (opponent at {opponent_pos})")
    
    plt.tight_layout()
    plt.savefig("vector_fields.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved vector_fields.png")


def parse_position(s):
    """Parse 'x,y' string into (float, float) tuple."""
    x, y = s.split(',')
    return (float(x), float(y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dog Game (Nash-Q)")
    parser.add_argument("--iterations", type=int, default=8000, help="Training iterations")
    parser.add_argument("--step-size", type=float, default=0.1, help="Movement step size")
    parser.add_argument("--horizon", type=int, default=10, help="Episode length")
    parser.add_argument("--house1", type=str, default="0.25,0.25", help="House 1 position (x,y)")
    parser.add_argument("--house2", type=str, default="0.75,0.75", help="House 2 position (x,y)")
    args = parser.parse_args()
    
    HORIZON = args.horizon
    HOUSE1 = parse_position(args.house1)
    HOUSE2 = parse_position(args.house2)
    
    env = DogGame(step_size=args.step_size, house1=HOUSE1, house2=HOUSE2)
    
    # Train
    nets, losses = neural_planning(env, iterations=args.iterations)
    net1, net2 = nets
    losses1, losses2 = losses
    policy_fn = get_policy(nets, env)
    
    # Export weights
    export_weights(net1, "weights_player1.txt", "Player 1 Q-network (Dog Game)")
    export_weights(net2, "weights_player2.txt", "Player 2 Q-network (Dog Game)")
    
    # Draw vector fields
    draw_vector_field(nets, env)
    
    # Plot loss
    plt.figure()
    plt.plot(losses1, label="P1 Loss", alpha=0.7)
    plt.plot(losses2, label="P2 Loss", alpha=0.7)
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Dog Game - Nash-Q Training Loss")
    plt.legend()
    plt.savefig("planning_loss.png")
    plt.close()
    print("Saved planning_loss.png")
    
    # Visualization
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    is_headless = plt.get_backend().lower() == "agg" or not has_display
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if is_headless:
        print(f"Headless mode. Saving 5 rollouts...")
        for i in range(5):
            s0 = env.sample_state()
            traj = rollout(env, s0, policy_fn)
            draw_trajectory(ax, traj)
            
            # Compute final distances
            final_dog = dog_position(traj[-1])
            d1 = distance(final_dog, HOUSE1)
            d2 = distance(final_dog, HOUSE2)
            
            ax.set_title(f"Rollout {i+1}: Dog to H1={d1:.3f}, to H2={d2:.3f}")
            fig.savefig(f"rollout_{i+1}.png", dpi=150, bbox_inches="tight")
            print(f"  Saved rollout_{i+1}.png | Start: ({s0[0]:.2f},{s0[1]:.2f}), ({s0[2]:.2f},{s0[3]:.2f})")
    else:
        from matplotlib.widgets import Button
        
        current_state = [env.sample_state()]
        
        def update_plot():
            s0 = current_state[0]
            traj = rollout(env, s0, policy_fn)
            draw_trajectory(ax, traj)
            
            final_dog = dog_position(traj[-1])
            d1 = distance(final_dog, HOUSE1)
            d2 = distance(final_dog, HOUSE2)
            
            fig.suptitle(f"Dog Game Rollout", fontsize=14)
            ax.set_title(f"Dog to House1: {d1:.3f} | Dog to House2: {d2:.3f}", fontsize=10)
            fig.canvas.draw_idle()
        
        plt.subplots_adjust(bottom=0.15)
        ax_new = plt.axes([0.4, 0.02, 0.2, 0.05])
        btn = Button(ax_new, "New Random")
        
        def new_state(event):
            current_state[0] = env.sample_state()
            update_plot()
        
        btn.on_clicked(new_state)
        update_plot()
        plt.show()
