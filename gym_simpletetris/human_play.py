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

    # Initialize Pygame and create a window
    pygame.init()
    window_size = (400, 500)  # Adjust as needed

    input_handler = HumanInputHandler(env.action_space, record_actions=record_actions)
    ui_manager = UIManager(*window_size)

    ui_manager.add_component(HeldPiece(10, 120, 100, 100))
    screen = pygame.display.set_mode(window_size)
    clock = pygame.time.Clock()

    done = False
    logic_updates_per_second = 10
    time_per_update = 1.0 / logic_updates_per_second
    last_logic_time = time.time()
    pause_duration = 1.0  # Pause for 1 second after a line clear
    paused = False
    pause_end_time = 0

    while not done:
        current_time = time.time()
        elapsed_time = current_time - last_logic_time

        # Process Pygame events every frame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

        # Handle input every frame
        action = input_handler.get_action(observation)
        if action == "quit":
            done = True
            break

        # Update game logic at fixed intervals
        if not paused and (current_time - last_logic_time) >= time_per_update:
            observation, reward, terminated, truncated, info = env.step(action)
            ui_manager.update(info)
            # Handle game over or reset
            if terminated or truncated:
                observation, info = env.reset()

            # Handle line clear pause
            cleared_lines = info.get("cleared_lines", 0)
            if cleared_lines > 0:
                paused = True
                pause_end_time = current_time + pause_duration
                print(
                    f"Paused for {pause_duration} seconds after clearing {cleared_lines} lines."
                )

            last_logic_time = current_time

        # Check if pause duration is over
        if paused and current_time >= pause_end_time:
            paused = False

        # Render the game state every frame
        board_array = env.render()

        # Convert the numpy array to a Pygame surface
        if board_array is not None:
            board_surface = pygame.surfarray.make_surface(board_array.swapaxes(0, 1))
            screen.fill((0, 0, 0))  # Fill with black
            screen.blit(board_surface, (0, 0))

        # Draw UI components
        ui_manager.draw(screen)

        # Update the display
        pygame.display.flip()

        # Optional: Control the frame rate (remove or adjust as needed)
        # clock.tick(60)  # Cap the FPS to 60 if desired

    env.close()
    input_handler.close()
    pygame.quit()
