import time
import gymnasium as gym
import gym_simpletetris
from gym_simpletetris.tetris.human_input_handler import HumanInputHandler


def main():
    env = gym.make("SimpleTetris-v0", render_mode="human")
    observation, info = env.reset()
    input_mode = "human"

    if input_mode == "human":
        input_handler = HumanInputHandler(env.action_space, record_actions=False)
    else:
        pass  # AI logic

    done = False
    logic_updates_per_second = 60
    time_per_update = 1.0 / logic_updates_per_second
    last_time = time.time()
    lag = 0.0
    pause_duration = 1.0  # Pause for 1 second after a line clear
    paused = False
    pause_end_time = 0

    while not done:
        current_time = time.time()
        elapsed_time = current_time - last_time
        last_time = current_time
        lag += elapsed_time

        if paused and current_time < pause_end_time:
            # Skip game updates and rendering during the pause
            continue
        elif paused and current_time >= pause_end_time:
            # End the pause and resume the game
            paused = False

        # Game logic update
        while lag >= time_per_update:
            action = input_handler.get_action(observation)
            if action == "quit":
                done = True
                break

            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Access cleared_lines from info dictionary
            cleared_lines = info.get("cleared_lines", 0)

            # Pause the game if lines were cleared
            if cleared_lines > 0:
                paused = True
                pause_end_time = current_time + pause_duration
                print(
                    f"Paused for {pause_duration} seconds after clearing {cleared_lines} lines."
                )

            lag -= time_per_update

        # Render at the end of the loop
        env.render()

    env.close()
    input_handler.close()

    if isinstance(input_handler, HumanInputHandler) and input_handler.record_actions:
        with open("training_data.pkl", "wb") as f:
            import pickle

            pickle.dump(input_handler.actions, f)


if __name__ == "__main__":
    main()
