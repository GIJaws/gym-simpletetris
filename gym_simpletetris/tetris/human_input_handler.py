from .input_handler import InputHandler
import pygame


class HumanInputHandler(InputHandler):
    def __init__(self, action_space, record_actions=False):
        self.action_space = action_space
        self.record_actions = record_actions
        self.actions = []
        pygame.init()
        pygame.display.set_caption("Human Tetris")

        # Updated key action map
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
        action = 7  # Default action is 'idle'

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

        keys = pygame.key.get_pressed()

        # Priority order: movement, rotation, other actions
        if keys[pygame.K_a]:
            action = self.key_action_map[pygame.K_a]
        elif keys[pygame.K_d]:
            action = self.key_action_map[pygame.K_d]

        # Allow rotation while moving
        if keys[pygame.K_LEFT]:
            action = 8 if action in [0, 1] else 4  # 8: Move+RotateLeft, 4: RotateLeft
        elif keys[pygame.K_RIGHT]:
            action = 9 if action in [0, 1] else 5  # 9: Move+RotateRight, 5: RotateRight

        # Other actions
        if keys[pygame.K_w]:
            action = self.key_action_map[pygame.K_w]
        elif keys[pygame.K_s]:
            action = self.key_action_map[pygame.K_s]
        elif keys[pygame.K_LSHIFT]:
            action = self.key_action_map[pygame.K_LSHIFT]
        elif keys[pygame.K_ESCAPE]:
            return "quit"

        if self.record_actions:
            self.actions.append((observation, action))

        return action

    def close(self):
        pygame.quit()
