import gymnasium as gym
import gym_simpletetris
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision.models import vit_b_16, ViT_B_16_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 1024
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 100000
TARGET_UPDATE = 1000
MEMORY_SIZE = 1000000
LEARNING_RATE = 1e-4
NUM_EPISODES = 100000
NUM_ENVS = 1  # Number of parallel environments


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input):
        if self.training:
            return F.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class TonyStarkDQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(TonyStarkDQN, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.vit.heads = nn.Identity()  # Remove the classification head

        encoder_layer = TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)

        self.advantage_stream = nn.Sequential(
            NoisyLinear(768, 512), nn.ReLU(), NoisyLinear(512, 256), nn.ReLU(), NoisyLinear(256, outputs)
        )

        self.value_stream = nn.Sequential(
            NoisyLinear(768, 512), nn.ReLU(), NoisyLinear(512, 256), nn.ReLU(), NoisyLinear(256, 1)
        )

    def forward(self, x):
        print("Input to ViT shape:", x.shape)
        x = self.vit(x)
        x = x.unsqueeze(0)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)

        advantage = self.advantage_stream(x)
        value = self.value_stream(x)
        return value + advantage - advantage.mean()


class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, *args):
        max_priority = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(args)
        else:
            self.memory[self.position] = args
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.position]

        probs = prios**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        beta = self.beta
        self.beta = min(1.0, self.beta + 0.001)

        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, device=device, dtype=torch.float)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)


def select_action(state, policy_net, steps_done, n_actions):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # Remove batch dimension for single state
            q_values = policy_net(state)
            return q_values.max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions, indices, weights = memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))

    # Concatenate states and next_states along the batch dimension
    state_batch = torch.cat(batch[0], dim=0)
    next_state_batch = torch.cat(batch[3], dim=0)
    print("state_batch.shape:", state_batch.shape)  # Should be (batch_size, 3, 224, 224)

    # Concatenate other tensors appropriately
    action_batch = torch.cat(batch[1], dim=0)  # Shape: (batch_size, 1)
    reward_batch = torch.stack(batch[2])  # Shape: (batch_size)
    done_batch = torch.cat(batch[4], dim=0)  # Shape: (batch_size)

    # Compute Q values
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute next state values
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[~done_batch] = target_net(next_state_batch[~done_batch]).max(1)[0]

    # Compute expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss
    loss = (state_action_values.squeeze() - expected_state_action_values).pow(2) * weights
    prios = loss + 1e-5
    loss = loss.mean()

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 100)
    optimizer.step()

    memory.update_priorities(indices, prios.data.cpu().numpy())


def train_dqn():
    envs = [gym.make("SimpleTetris-v0", render_mode=None if i == 0 else None) for i in range(NUM_ENVS)]
    n_actions = envs[0].action_space.n

    initial_state, _ = envs[0].reset()
    state = simplify_board(initial_state)
    h, w = state.shape

    # Correct the tensor shape for ViT input
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
    state = state.repeat(1, 3, 1, 1)
    state = F.interpolate(state, size=(224, 224), mode="bilinear", align_corners=False)

    policy_net = TonyStarkDQN(224, 224, n_actions).to(device)
    target_net = TonyStarkDQN(224, 224, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = PrioritizedReplayMemory(MEMORY_SIZE)

    steps_done = 0
    episode_rewards = []

    for episode in range(NUM_EPISODES):
        states = [env.reset()[0] for env in envs]
        states = [
            F.interpolate(
                torch.tensor(simplify_board(state), dtype=torch.float32, device=device)
                .unsqueeze(0)  # Add batch dimension
                .unsqueeze(1)  # Add channel dimension
                .repeat(1, 3, 1, 1),  # Repeat channels to simulate RGB
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            for state in states
        ]

        total_rewards = [0] * NUM_ENVS
        dones = [False] * NUM_ENVS

        while not all(dones):
            actions = [select_action(state, policy_net, steps_done, n_actions) for state in states]
            steps_done += NUM_ENVS

            next_states, rewards, terminated, truncated, infos = zip(
                *[env.step([action.item()]) for env, action in zip(envs, actions)]
            )
            rewards = [
                calculate_reward(next_state, info["lines_cleared"], term)
                for next_state, info, term in zip(next_states, infos, terminated)
            ]
            next_states = [
                F.interpolate(
                    torch.tensor(simplify_board(state), dtype=torch.float32, device=device)
                    .unsqueeze(0)
                    .unsqueeze(1)
                    .repeat(1, 3, 1, 1),
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
                for state in next_states
            ]

            rewards = torch.tensor(rewards, device=device)
            dones = [term or trunc for term, trunc in zip(terminated, truncated)]

            for i in range(NUM_ENVS):
                memory.push(states[i], actions[i], rewards[i], next_states[i], torch.tensor([dones[i]], device=device))
                total_rewards[i] += rewards[i]

            states = next_states

            optimize_model(memory, policy_net, target_net, optimizer)

        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.extend(total_rewards)
        avg_reward = np.mean(
            [reward.cpu().item() if isinstance(reward, torch.Tensor) else reward for reward in episode_rewards[-100:]]
        )

        print(f"Episode {episode + 1}/{NUM_EPISODES}, Avg Reward (last 100): {avg_reward:.2f}")
        print(f"Current VRAM usage: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")

        if (episode + 1) % 100 == 0:
            torch.save(policy_net.state_dict(), f"tetris_tony_stark_dqn_episode_{episode + 1}.pth")

    for env in envs:
        env.close()
    print("Training completed")


def calculate_reward(board, lines_cleared, game_over):
    if game_over:
        return -1

    if isinstance(board, torch.Tensor):
        board = board.squeeze().cpu().numpy()

    # Ensure we're working with a 2D array
    if board.ndim == 3:
        board = np.any(board != 0, axis=2)

    heights = [0] * board.shape[1]
    for i in range(board.shape[1]):
        for j in range(board.shape[0]):
            if board[j, i]:  # Changed from board[j, i] != 0
                heights[i] = board.shape[0] - j
                break

    max_height = max(heights)
    holes = 0
    for i in range(board.shape[1]):
        holes += len(
            [x for x in range(heights[i]) if not board[board.shape[0] - x - 1, i]]
        )  # Changed from board[board.shape[0] - x - 1, i] == 0

    bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))

    height_weight = -0.51
    hole_weight = -0.35
    bumpiness_weight = -0.18
    lines_cleared_weight = 1

    reward = max_height * height_weight
    reward += holes * hole_weight
    reward += bumpiness * bumpiness_weight
    reward += lines_cleared * lines_cleared_weight

    return reward


def simplify_board(board):
    if board.ndim == 3:
        return np.any(board != 0, axis=2).astype(np.float32)
    return board.astype(np.float32)


if __name__ == "__main__":
    train_dqn()
