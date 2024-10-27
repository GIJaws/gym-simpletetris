from abc import ABC, abstractmethod
from typing import Any


class InputHandler(ABC):
    @abstractmethod
    def get_action(self, observation) -> Any:
        pass

    def close(self):
        pass
