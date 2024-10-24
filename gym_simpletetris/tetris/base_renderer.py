import numpy as np
from abc import ABC, abstractmethod

from gym_simpletetris.tetris.tetris_engine import GameState


class BaseRenderer(ABC):
    def __init__(self, width, height, obs_type, **kwargs):
        self.width: int = width
        self.height: int = height
        self.obs_type = obs_type

    @abstractmethod
    def render(self, game_state: GameState) -> np.ndarray | None:
        pass

    @abstractmethod
    def close(self):
        pass
