import gymnasium as gym
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
import gym_simpletetris
from gym_simpletetris.tetris.human_input_handler import HumanInputHandler


def play_tetris(render_mode="human", record_actions=False):

    env = gym.make("SimpleTetris-v0", render_mode=render_mode, num_lives=1000, render_fps=5)
    observation, info = env.reset()
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

        if game_over:
            print(f"Game over! Final Score: {info['score']}")
            done = True

    env.close()
    input_handler.close()


if __name__ == "__main__":
    play_tetris()
