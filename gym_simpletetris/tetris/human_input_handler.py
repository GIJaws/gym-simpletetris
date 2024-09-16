from .input_handler import InputHandler
import pygame


class HumanInputHandler(InputHandler):
    def __init__(self, action_space, record_actions=False):
        self.action_space = action_space
        self.record_actions = record_actions
        self.actions = []
        self.current_action = None  # Store the current action
        # Initialize Pygame keys
        self.key_action_map = {
            pygame.K_a: 0,  # Move Left
            pygame.K_d: 1,  # Move Right
            pygame.K_s: 3,  # Soft Drop
            pygame.K_w: 2,  # Hard Drop
            pygame.K_LEFT: 4,  # Rotate Left
            pygame.K_RIGHT: 5,  # Rotate Right
            pygame.K_LSHIFT: 6,  # Hold/Swap
            pygame.K_ESCAPE: "quit",
        }

    def get_action(self, observation):
        # Get the state of all keyboard buttons
        keys = pygame.key.get_pressed()

        # Determine action based on keys pressed
        for key, action in self.key_action_map.items():
            if keys[key]:
                if action == "quit":
                    return "quit"
                else:
                    self.current_action = action
                    break
        else:
            self.current_action = 7  # Default to 'idle' if no key is pressed

        if self.record_actions:
            self.actions.append((observation, self.current_action))

        return self.current_action

    def close(self):
        pass  # Do not quit Pygame here
