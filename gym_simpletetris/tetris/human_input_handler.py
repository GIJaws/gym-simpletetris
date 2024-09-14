import pygame
from .input_handler import InputHandler


class HumanInputHandler(InputHandler):
    def __init__(self, action_space, record_actions=False):
        self.action_space = action_space
        self.record_actions = record_actions
        self.actions = []
        pygame.init()
        pygame.display.set_caption("Human Tetris")

        # Updated key action map based on your requirements
        self.key_action_map = {
            pygame.K_a: 0,  # Move Left (A)
            pygame.K_d: 1,  # Move Right (D)
            pygame.K_s: 3,  # Soft Drop (S)
            pygame.K_w: 2,  # Hard Drop (W)
            pygame.K_LEFT: 4,  # Rotate Left (Arrow Left)
            pygame.K_RIGHT: 5,  # Rotate Right (Arrow Right)
            pygame.K_LSHIFT: 6,  # Hold/Swap (Shift)
            pygame.K_ESCAPE: "quit",
        }

    def get_action(self, observation):
        action = 7  # Default action is 'idle' (assuming 7 is idle)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

        keys = pygame.key.get_pressed()  # Detect which keys are held down

        # Handle movement based on held-down keys for WASD and Arrow keys
        if keys[pygame.K_a]:  # Move Left (A)
            action = self.key_action_map[pygame.K_a]
        elif keys[pygame.K_d]:  # Move Right (D)
            action = self.key_action_map[pygame.K_d]
        elif keys[pygame.K_s]:  # Soft Drop (S)
            action = self.key_action_map[pygame.K_s]
        elif keys[pygame.K_w]:  # Hard Drop (W)
            action = self.key_action_map[pygame.K_w]
        elif keys[pygame.K_LEFT]:  # Rotate Left (Arrow Left)
            action = self.key_action_map[pygame.K_LEFT]
        elif keys[pygame.K_RIGHT]:  # Rotate Right (Arrow Right)
            action = self.key_action_map[pygame.K_RIGHT]
        elif keys[pygame.K_LSHIFT]:  # Hold/Swap (Shift)
            action = self.key_action_map[pygame.K_LSHIFT]
        elif keys[pygame.K_ESCAPE]:
            return "quit"

        if self.record_actions:
            self.actions.append((observation, action))

        return action

    def close(self):
        pygame.quit()
