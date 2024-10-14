import numpy as np
from dataclasses import dataclass
from enum import Enum
import random


@dataclass(frozen=True)
class Piece:
    name: str
    shape: np.ndarray
    max_orientation: int
    position: tuple[int, int] = (-1, -1)  # default to invalid position
    orientation: int = 0  # 0 to 3 for the four possible orientations
    color: tuple[int, int, int] = (0, 0, 0)  # default color


class PieceType(Enum):
    T = Piece("T", np.array(((0, 0), (-1, 0), (1, 0), (0, -1))), 4, color=(128, 0, 128))  # Purple
    J = Piece("J", np.array(((0, 0), (-1, 0), (0, -1), (0, -2))), 4, color=(0, 255, 0))  # Blue
    L = Piece("L", np.array(((0, 0), (1, 0), (0, -1), (0, -2))), 4, color=(165, 0, 255))  # Orange
    Z = Piece("Z", np.array(((0, 0), (-1, 0), (0, -1), (1, -1))), 2, color=(0, 0, 255))  # Red
    S = Piece("S", np.array(((0, 0), (-1, -1), (0, -1), (1, 0))), 2, color=(255, 0, 0))  # Green
    I = Piece("I", np.array(((0, 0), (0, -1), (0, -2), (0, -3))), 2, color=(255, 255, 0))  # Cyan
    O = Piece("O", np.array(((0, 0), (0, -1), (-1, 0), (-1, -1))), 1, color=(0, 255, 255))  # Yellow

    @staticmethod
    def piece_names() -> list[str]:
        return [piece_type.name for piece_type in PieceType]


class PieceQueue:
    def __init__(self, preview_size=4):
        self.preview_size = max(1, preview_size)  # Ensure at least 1 piece in preview
        self.pieces: list[Piece] = []
        self.bag: list[Piece] = []
        self.refill_bag()
        self.fill_queue()

    def refill_bag(self):
        self.bag = list(foo.value for foo in PieceType)
        random.shuffle(self.bag)

    def next_piece(self) -> Piece:
        if len(self.pieces) <= self.preview_size:
            self.fill_queue()
        return self.pieces.pop(0)

    def fill_queue(self):
        while len(self.pieces) < self.preview_size * 2:  # Keep 2x preview size in queue
            if not self.bag:
                self.refill_bag()
            self.pieces.append(self.bag.pop())

    def get_preview(self):
        while len(self.pieces) < self.preview_size:
            self.fill_queue()
        return self.pieces[: self.preview_size]
