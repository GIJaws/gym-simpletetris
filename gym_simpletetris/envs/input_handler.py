# gym_simpletetris/envs/input_handler.py

from abc import ABC, abstractmethod


class InputHandler(ABC):
    @abstractmethod
    def get_action(self, game_state):
        pass


class RandomInputHandler(InputHandler):
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, game_state):
        return self.action_space.sample()


# You can add more input handlers here in the future, like:
# class HumanInputHandler(InputHandler):
#     def get_action(self, game_state):
#         # Implement keyboard input logic here
#         pass
