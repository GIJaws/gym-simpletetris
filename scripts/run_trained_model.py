import gymnasium as gym
import gym_simpletetris
import torch
import numpy as np
from run_torch_ai_v2 import DQN, simplify_board  # Import from your training script

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def play_tetris(model_path, num_games=5):
    env = gym.make("SimpleTetris-v0", render_mode="human")

    # Get the shape of the board
    initial_state, _ = env.reset()
    simplified_state = simplify_board(initial_state)
    h, w = simplified_state.shape
    n_actions = env.action_space.n

    # Load the trained model
    model = DQN(h, w, n_actions).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode

    for game in range(num_games):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            # Prepare the state for the model
            state_simplified = simplify_board(state)
            state_tensor = torch.tensor(state_simplified, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

            # Get the action from the model
            with torch.no_grad():
                action = model(state_tensor).max(1)[1].view(1, 1)

            # Take the action in the environment
            state, reward, terminated, truncated, info = env.step([action.item()])
            total_reward += reward
            steps += 1
            done = terminated or truncated

            # Optional: Add a small delay to make the game visible
            env.render()
            # time.sleep(0.1)  # Uncomment this line to slow down the game

        print(f"Game {game + 1} finished. Score: {total_reward}, Steps: {steps}")

    env.close()


if __name__ == "__main__":
    model_path = "tetris_dqn_episode_1100.pth"  # Update this to your latest model file
    play_tetris(model_path)
