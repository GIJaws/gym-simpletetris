import itertools
from .input_handler import InputHandler
import pygame
import logging

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


class HumanInputHandler(InputHandler):
    def __init__(self, action_space, record_actions=False):
        self.action_space = action_space
        self.record_actions = record_actions
        self.actions = []

        # Initialize Pygame keys and actions
        self.key_action_map = {
            pygame.K_a: 0,  # Move Left
            pygame.K_d: 1,  # Move Right
            pygame.K_w: 2,  # Hard Drop
            pygame.K_s: 3,  # Soft Drop
            pygame.K_LEFT: 4,  # Rotate Left
            pygame.K_RIGHT: 5,  # Rotate Right
            pygame.K_LSHIFT: 6,  # Hold/Swap
            pygame.K_ESCAPE: "quit",
        }

    def get_action(self, observation):
        keys = pygame.key.get_pressed()

        # Only log if any relevant keys are pressed
        # relevant_keys_pressed = any(keys[key] for key in self.key_action_map)

        # if relevant_keys_pressed:
        #     logging.info(f"Keys pressed: {[pygame.key.name(key) for key in self.key_action_map if keys[key]]}")

        actions = [action for key, action in self.key_action_map.items() if keys[key]] or [7]
        # logging.info(f"Current action set: {actions}")

        if "quit" in actions:
            # logging.info("Quit action detected")
            return "quit"

        if self.record_actions:
            self.actions.append((observation, actions))

        return actions

    def close(self):
        pass  # Do not quit Pygame here
