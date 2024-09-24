from abc import ABC, abstractmethod
import numpy as np


class AbstractScoringSystem(ABC):
    @abstractmethod
    def calculate_step_reward(self) -> float:
        pass

    @abstractmethod
    def calculate_clear_reward(self, cleared_lines: int) -> float:
        pass

    @abstractmethod
    def calculate_height_penalty(self, board: np.ndarray) -> float:
        pass

    @abstractmethod
    def calculate_height_increase_penalty(self, new_height: int, old_height: int) -> float:
        pass

    @abstractmethod
    def calculate_holes_penalty(self, holes: int) -> float:
        pass

    @abstractmethod
    def calculate_holes_increase_penalty(self, new_holes: int, old_holes: int) -> float:
        pass


class ScoringSystem(AbstractScoringSystem):
    def __init__(self, config: dict[str, bool]):
        self.config = config

    def calculate_step_reward(self) -> float:
        return 1 if self.config.get("reward_step", False) else 0

    def calculate_clear_reward(self, cleared_lines: int) -> float:
        if self.config.get("advanced_clears", False):
            scores = [0, 40, 100, 300, 1200]
            return 2.5 * scores[cleared_lines]
        elif self.config.get("high_scoring", False):
            return 1000 * cleared_lines
        else:
            return 100 * cleared_lines

    def calculate_height_penalty(self, board: np.ndarray) -> float:
        if self.config.get("penalise_height", False):
            return -sum(any(board[:, i]) for i in range(board.shape[1]))
        return 0

    def calculate_height_increase_penalty(self, new_height: int, old_height: int) -> float:
        if self.config.get("penalise_height_increase", False) and new_height > old_height:
            return -10 * (new_height - old_height)
        return 0

    def calculate_holes_penalty(self, holes: int) -> float:
        if self.config.get("penalise_holes", False):
            return -5 * holes
        return 0

    def calculate_holes_increase_penalty(self, new_holes: int, old_holes: int) -> float:
        if self.config.get("penalise_holes_increase", False):
            return -5 * (new_holes - old_holes)
        return 0
