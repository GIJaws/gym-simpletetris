from abc import ABC, abstractmethod


class InputHandler(ABC):
    @abstractmethod
    def get_action(self, observation):
        pass

    def close(self):
        pass
