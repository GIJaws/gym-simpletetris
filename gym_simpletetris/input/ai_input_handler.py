from typing_extensions import deprecated
from gym_simpletetris.input.input_handler import InputHandler


@deprecated("Is anything even using this?")
class AIInputHandler(InputHandler):
    def __init__(self, agent):
        self.agent = agent

    def get_action(self, observation):
        action = self.agent.act(observation)
        return action
