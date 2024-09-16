import time
import pygame
import numpy as np
import gymnasium as gym
import gym_simpletetris
from gym_simpletetris.tetris.human_input_handler import HumanInputHandler
from gym_simpletetris.ui.ui import UIManager, PiecePreview, HeldPiece, ScoreDisplay


def play_tetris(render_mode="human", record_actions=False):
    env = gym.make("SimpleTetris-v0", render_mode=render_mode)
    observation, info = env.reset()

    input_handler = HumanInputHandler(env.action_space, record_actions=record_actions)
    ui_manager = UIManager()

    ui_manager.add_component(PiecePreview(10, 10, 100, 100))
    ui_manager.add_component(HeldPiece(10, 120, 100, 100))
    ui_manager.add_component(ScoreDisplay(10, 230, 100, 30))

    # Initialize Pygame and create a window
    pygame.init()
    window_size = (400, 500)  # Adjust as needed
    screen = pygame.display.set_mode(window_size)
    clock = pygame.time.Clock()

    done = False
    logic_updates_per_second = 10
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
            ui_manager.update(info)

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
            if terminated or truncated:
                observation, info = env.reset()

        # Render the game state
        board_array = env.render()

        # Convert the numpy array to a Pygame surface
        try:
            if board_array is not None:  # Check if board_array is not None
                board_surface = pygame.surfarray.make_surface(
                    board_array.swapaxes(0, 1)
                )
                screen.blit(board_surface, (0, 0))
        except ValueError as e:
            print(f"Error converting board array to surface: {e}")
            print(f"Array shape: {board_array.shape}, dtype: {board_array.dtype}")
            continue

        # # Clear the screen
        # screen.fill((0, 0, 0))  # Fill with black

        # # Draw the Tetris board
        # screen.blit(board_surface, (0, 0))

        # Draw UI components
        ui_manager.draw(screen)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)  # 60 FPS cap

    env.close()
    input_handler.close()
    pygame.quit()

    if isinstance(input_handler, HumanInputHandler) and input_handler.record_actions:
        with open("training_data.pkl", "wb") as f:
            import pickle

            pickle.dump(input_handler.actions, f)


if __name__ == "__main__":
    play_tetris()
