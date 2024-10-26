import numpy as np
from numpy.typing import NDArray
from pprint import pformat
from dataclasses import dataclass, replace
from enum import Enum
import random


@dataclass(frozen=True)
class Piece:
    name: str
    shape: NDArray[np.int8]
    max_orientation: int
    # TODO make position and color a named tuple
    position: tuple[np.uint8, np.uint8] = (np.uint8(0), np.uint8(0))
    orientation: int = 0  # 0 to 3 for the four possible orientations
    color: tuple[int, int, int] = (0, 0, 0)  # default color

    # def __post_init__(self):
    #     print("hellooo")
    #     print(self)

    def rotate(self, clockwise: bool = True) -> "Piece":
        """
        Rotate the piece clockwise or counterclockwise and return a new instance with the rotated shape,
        only if the max_orientation allows it.

        Parameters
        ----------
        clockwise : bool, optional
            If True, rotate the piece clockwise. If False, rotate counterclockwise. Defaults to True.

        Returns
        -------
        Piece
            A new Piece instance with the rotated shape, or the original piece if rotation is not allowed.
        """
        if self.max_orientation == 1:
            return self  # No rotation allowed

        rotation_matrix = np.array([[0, -1], [1, 0]]) if clockwise else np.array([[0, 1], [-1, 0]])
        rotated_shape = self.shape @ rotation_matrix

        new_orientation = (self.orientation + (1 if clockwise else -1)) % self.max_orientation

        # Return a new Piece instance with the rotated shape, keeping other attributes the same.
        return replace(self, shape=rotated_shape, orientation=new_orientation)

    def get_render_blocks(self):
        """
        Process piece data into a format suitable for rendering.

        Returns:
            list: A list of tuples (x, y, color) representing each block of the piece.
        """
        x, y = self.position
        blocks = [(x + i, y + j, self.color) for i, j in self.shape]
        return blocks

    def __str__(self):
        # Find the bounds of the piece
        min_x = min(x for x, _ in self.shape)
        max_x = max(x for x, _ in self.shape)
        min_y = min(y for _, y in self.shape)
        max_y = max(y for _, y in self.shape)

        # Create a 2D grid to represent the piece
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        grid = [[" " for _ in range(width)] for _ in range(height)]

        # Fill in the grid
        for x, y in self.shape:
            grid[y - min_y][x - min_x] = "■"

        # Convert the grid to a string
        # piece_shape = "\nPretty Shape:\n" + "\n".join(" ".join(row) for row in reversed(grid))
        piece_shape = "\nPretty Shape:\n" + "\n".join(" ".join(row) for row in grid)

        # Pretty print the shape array using pprint
        pretty_shape = pformat(self.shape.tolist(), width=100)

        # Add piece information
        info = f"Name: {self.name}, Position: {self.position}, Orientation: {self.orientation}\nShape: {pretty_shape}"

        return info + piece_shape


class PieceType(Enum):
    T = Piece("T", np.array(((0, 0), (-1, 0), (1, 0), (0, -1))), 4, color=(128, 0, 128))  # Purple
    J = Piece("J", np.array(((0, 0), (-1, 0), (0, -1), (0, -2))), 4, color=(0, 255, 0))  # Blue
    L = Piece("L", np.array(((0, 0), (1, 0), (0, -1), (0, -2))), 4, color=(165, 0, 255))  # Orange
    Z = Piece("Z", np.array(((0, 0), (-1, -1), (0, -1), (1, 0))), 2, color=(0, 0, 255))  # Red
    S = Piece("S", np.array(((0, 0), (-1, 0), (0, -1), (1, -1))), 2, color=(255, 0, 0))  # Green
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
