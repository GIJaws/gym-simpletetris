# human_input_handler.py
import pygame
from .input_handler import InputHandler


class HumanInputHandler(InputHandler):
    def __init__(self, action_space, record_actions=False):
        self.action_space = action_space
        self.record_actions = record_actions
        self.actions = []
        pygame.init()
        pygame.display.set_caption("Human Tetris")
        self.key_action_map = {
            pygame.K_LEFT: 0,  # Left
            pygame.K_RIGHT: 1,  # Right
            pygame.K_DOWN: 3,  # Soft Drop
            pygame.K_UP: 5,  # Rotate Right
            pygame.K_z: 4,  # Rotate Left
            pygame.K_SPACE: 2,  # Hard Drop
            pygame.K_ESCAPE: "quit",
        }

    def get_action(self, observation):
        action = 6  # Default action is 'idle'
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.KEYDOWN:
                if event.key in self.key_action_map:
                    action = self.key_action_map[event.key]
                    if action == "quit":
                        return "quit"
        if self.record_actions:
            self.actions.append((observation, action))
        return action

    def close(self):
        pygame.quit()
