import gymnasium as gym
import gym_simpletetris
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 1000000  # Decay over 1 million steps
TARGET_UPDATE = 1000  # Update target network every 1000 steps
MEMORY_SIZE = 100000
LEARNING_RATE = 1e-4
NUM_EPISODES = 10000


# Define the neural network
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Assuming 3 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc_input_dim = 64 * h * w
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, outputs)

    def forward(self, x):
        x = x / 255.0  # Normalize input
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, self.fc_input_dim)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Select action based on epsilon-greedy policy
def select_action(state, policy_net, steps_done, n_actions):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * steps_done / EPS_DECAY)
    if random.random() < eps_threshold:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).view(1, 1)


# Optimize the model
def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

    batch_state = torch.cat(batch_state)
    batch_action = torch.cat(batch_action)
    batch_reward = torch.cat(batch_reward)
    batch_next_state = torch.cat(batch_next_state)
    batch_done = torch.cat(batch_done)

    # Compute Q(s_t, a)
    state_action_values = policy_net(batch_state).gather(1, batch_action)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[~batch_done] = target_net(batch_next_state[~batch_done]).max(1)[0].detach()

    # Compute expected Q values
    expected_state_action_values = batch_reward + (GAMMA * next_state_values)
    expected_state_action_values = expected_state_action_values.unsqueeze(1)

    # Compute Huber loss
    loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Main training loop
def train_dqn():
    env = gym.make("SimpleTetris-v0", render_mode="human")  # Set render_mode to 'human' to visualize
    n_actions = env.action_space.n

    init_screen, _ = env.reset()
    print(f"init_screen.shape: {init_screen.shape}")  # For debugging

    w, h, _ = init_screen.shape  # Corrected dimension assignment
    print(f"Width: {w}, Height: {h}")

    policy_net = DQN(h, w, n_actions).to(device)
    target_net = DQN(h, w, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        # Correct the transpose to match the expected dimensions
        state = torch.tensor(state.transpose((2, 1, 0)), dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        done = False

        while not done:
            action = select_action(state, policy_net, steps_done, n_actions)
            steps_done += 1

            next_state, reward, terminated, truncated, _ = env.step([action.item()])
            total_reward += reward
            done = terminated or truncated

            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            if not done:
                next_state_tensor = torch.tensor(
                    next_state.transpose((2, 1, 0)), dtype=torch.float32, device=device
                ).unsqueeze(0)
            else:
                next_state_tensor = torch.zeros_like(state, device=device)

            # Store the transition in memory
            memory.push(state, action, reward, next_state_tensor, torch.tensor([done], device=device))

            state = next_state_tensor

            # Perform one step of the optimization
            optimize_model(memory, policy_net, target_net, optimizer)

            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode + 1}/{NUM_EPISODES}, Total Reward: {total_reward}")
        # Optional: Save the model every N episodes
        if (episode + 1) % 100 == 0:
            torch.save(policy_net.state_dict(), f"tetris_dqn_episode_{episode + 1}.pth")

    env.close()
    print("Training completed")


if __name__ == "__main__":
    train_dqn()
