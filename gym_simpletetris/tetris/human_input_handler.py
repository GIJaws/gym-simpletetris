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
        self.current_action = None  # Store the current action

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

        # Automate action combination generation
        self.combination_action_map = self.generate_combinations(self.key_action_map)

    def generate_combinations(self, key_action_map):
        # Define action groups that can logically be combined
        movement_actions = [0, 1]  # Left, Right
        rotation_actions = [4, 5]  # Rotate Left, Rotate Right
        other_actions = [2, 3, 6]  # Hard Drop, Soft Drop, Hold/Swap

        # Create all valid combinations of these actions
        valid_combinations = list(
            itertools.chain(
                itertools.product(movement_actions, rotation_actions),
                itertools.product(movement_actions, other_actions),
                itertools.product(rotation_actions, other_actions),
            )
        )

        # Map combinations to unique action values
        combination_map = {}
        for idx, combo in enumerate(valid_combinations, start=8):
            combination_map[combo] = idx  # Start assigning actions from 8 upwards

        return combination_map

    def get_action(self, observation):
        keys = pygame.key.get_pressed()
        actions = []

        # Only log if any relevant keys are pressed
        relevant_keys_pressed = any(keys[key] for key in self.key_action_map.keys())

        if relevant_keys_pressed:
            logging.info(f"Keys pressed: {[pygame.key.name(key) for key in self.key_action_map if keys[key]]}")

        # Track which actions are being processed
        for key, action in self.key_action_map.items():
            if keys[key]:
                actions.append(action)

        if "quit" in actions:
            logging.info("Quit action detected")
            return "quit"

        # Check if multiple actions are pressed and map to a combination
        if len(actions) > 1:
            action_tuple = tuple(sorted(actions))
            if action_tuple in self.combination_action_map:
                combined_action = self.combination_action_map[action_tuple]
                logging.info(f"Combination action detected: {combined_action}")
                self.current_action = combined_action
            else:
                # If no predefined combination, pick the first action
                logging.info(f"Multiple actions detected: {actions}")
                self.current_action = actions[0]
        elif actions:
            self.current_action = actions[0]
            logging.info(f"Current action set: {self.current_action}")
        else:
            if relevant_keys_pressed:  # Only log idle if keys were pressed
                logging.info("No valid action detected, setting to idle.")
            self.current_action = 7  # Idle action

        if self.record_actions:
            self.actions.append((observation, self.current_action))
            logging.debug(f"Recorded action: {self.current_action}")

        return self.current_action

    def close(self):
        pass  # Do not quit Pygame here
