import numpy as np
from abc import ABC, abstractmethod


class BaseRenderer(ABC):
    def __init__(self, width, height, obs_type, **kwargs):
        self.width = width
        self.height = height
        self.obs_type = obs_type

    @abstractmethod
    def render(self, game_state) -> np.ndarray | None:
        pass

    @abstractmethod
    def close(self):
        pass
