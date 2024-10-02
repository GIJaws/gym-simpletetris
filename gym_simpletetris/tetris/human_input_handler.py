from .input_handler import InputHandler
import pygame


class HumanInputHandler(InputHandler):
    def __init__(self, action_space, record_actions=False):
        self.action_space = action_space
        self.record_actions = record_actions
        self.actions = []

        self.cooldowns = {k: 0 for k in range(8)}  # Cooldown for each action
        self.das_delay = 8  # ~13.5ms at 60fps
        self.arr_delay = 3  # ~50 at 60fps
        self.das_timers = {}
        self.pressed_keys = set()

        # Initialize Pygame keys and actions
        self.key_action_map = {
            pygame.K_a: 0,  # Move Left
            pygame.K_d: 1,  # Move Right
            pygame.K_LEFT: 2,  # Rotate Left
            pygame.K_RIGHT: 3,  # Rotate Right
            pygame.K_LSHIFT: 4,  # Hold/Swap
            pygame.K_w: 5,  # Hard Drop
            pygame.K_s: 6,  # Soft Drop
            pygame.K_ESCAPE: "quit",
        }

        pygame.key.set_repeat(500, 100)

    def get_action(self, observation):
        actions = set()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key not in self.pressed_keys:
                    self.pressed_keys.add(event.key)
                    action = self.key_action_map.get(event.key)
                    if action is not None:
                        if action == "quit":
                            return "quit"
                        actions.add(action)
                        self.das_timers[event.key] = 0
            elif event.type == pygame.KEYUP:
                if event.key in self.pressed_keys:
                    self.pressed_keys.remove(event.key)
                if event.key in self.das_timers:
                    del self.das_timers[event.key]

        # Handle DAS (Delayed Auto Shift)
        keys = pygame.key.get_pressed()
        for key, action in self.key_action_map.items():
            if keys[key] and key in self.das_timers:
                if self.das_timers[key] >= self.das_delay:
                    if self.cooldowns[action] == 0:
                        actions.add(action)
                        self.cooldowns[action] = self.arr_delay
                self.das_timers[key] += 1

        # Apply cooldowns
        for action in range(8):
            if self.cooldowns[action] > 0:
                self.cooldowns[action] -= 1

        actions = list(actions) or [7]  # Default to IDLE if no action

        if self.record_actions:
            self.actions.append((observation, actions))

        return actions

    def close(self):
        pass  # Do not quit Pygame here
