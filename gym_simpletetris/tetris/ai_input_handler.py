# ai_input_handler.py
from .input_handler import InputHandler


class AIInputHandler(InputHandler):
    def __init__(self, agent):
        self.agent = agent

    def get_action(self, observation):
        action = self.agent.act(observation)
        return action
