import gymnasium as gym
import numpy as np
import gym_simpletetris
from gym_simpletetris.tetris.human_input_handler import HumanInputHandler


def calculate_reward(board, lines_cleared, game_over, last_board, prev_lines_cleared):
    # Ensure we're working with a 2D array of booleans
    if board.ndim == 3:
        board = np.any(board != 0, axis=2).astype(bool)
    if last_board.ndim == 3:
        last_board = np.any(last_board != 0, axis=2).astype(bool)

    reward = 0

    # Game Over Penalty
    if game_over:
        reward += -100
        return reward  # End the game early with the penalty for game over

    # Line Clear Reward
    line_clear_reward = [0, 40, 100, 300, 1200]  # Standard Tetris scoring
    reward += line_clear_reward[lines_cleared - prev_lines_cleared]

    # Height Calculation
    heights = np.array([board.shape[0] - np.argmax(column) if np.any(column) else 0 for column in board.T])

    # Stack Height Penalty
    max_height = np.max(heights)
    reward += -0.5 * max_height

    # Holes Calculation
    holes = 0
    for x in range(board.shape[1]):
        column = board[:, x].astype(bool)  # Ensure the column is boolean
        filled = np.where(column)[0]  # Get indices of filled cells
        if filled.size > 0:
            # Count empty cells below the first filled cell
            holes += np.sum(~column[filled[0] :])

    reward += -0.7 * holes

    # Bumpiness Calculation
    bumpiness = np.sum(np.abs(np.diff(heights)))
    reward += -0.2 * bumpiness

    # No need for idling penalty in human play

    return reward


def play_tetris(render_mode="human", record_actions=False):

    env = gym.make("SimpleTetris-v0", render_mode=render_mode)
    observation, info = env.reset()
    last_observation = np.copy(observation)
    prev_lines_cleared = 0
    input_handler = HumanInputHandler(env.action_space, record_actions=record_actions)

    done = False
    action = [7]
    while not done:

        observation, reward, terminated, truncated, info = env.step(action)
        action = input_handler.get_action(observation)
        if action == "quit":
            done = True
            break
        # Calculate reward based on game state
        game_over = terminated or truncated
        current_reward = calculate_reward(
            observation, info["lines_cleared"], game_over, last_observation, prev_lines_cleared
        )

        # Print the current reward to the console
        print(f"Reward: {current_reward}, Lines Cleared: {info['lines_cleared']}")

        # Update last_observation for the next iteration
        last_observation = np.copy(observation)
        prev_lines_cleared = info["lines_cleared"]
        if game_over:
            print(f"Game over! Final Score: {info['score']}")
            done = True

    env.close()
    input_handler.close()


if __name__ == "__main__":
    play_tetris()
