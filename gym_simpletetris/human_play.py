import gymnasium as gym
import gym_simpletetris
from gym_simpletetris.tetris.human_input_handler import HumanInputHandler


def play_tetris(render_mode="human", record_actions=False):

    env = gym.make("SimpleTetris-v0", render_mode=render_mode)
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

        env.render()  # This line renders the current state

        if terminated or truncated:
            print(f"Game over! Score: {info['score']}")
            done = True

    env.close()
    input_handler.close()


if __name__ == "__main__":
    play_tetris()
