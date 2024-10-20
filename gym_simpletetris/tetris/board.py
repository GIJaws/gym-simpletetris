import numpy as np
from numpy.typing import NDArray
from gym_simpletetris.tetris.pieces import Piece
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Board:
    buffer_height: int
    width: int
    height: int
    grid: NDArray[np.uint8]  # assumption is that this is an immutable 2D grid
    spawn_y: np.uint8 = field(init=False)
    total_height: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "spawn_y", np.uint8(self.buffer_height - 1))
        object.__setattr__(self, "total_height", self.height + self.buffer_height)

    @staticmethod
    def create_board(width: int, height: int, buffer_height: int) -> "Board":
        grid = np.packbits(np.zeros((width, (height + buffer_height)), dtype=np.uint8))
        return Board(buffer_height=buffer_height, width=width, height=height, grid=grid)

    def place_piece(self, piece: Piece, block: bool = True) -> "Board":
        new_grid = self.grid.copy()
        for x_offset, y_offset in piece.shape:
            x = piece.position[0] + x_offset
            y = piece.position[1] + y_offset
            if (0 <= x < self.width) and (0 <= y < (self.total_height)):
                byte_index = (x * (self.total_height) + y) // 8
                bit_index = (x * (self.total_height) + y) % 8
                if block:
                    new_grid[byte_index] |= 1 << (7 - bit_index)
                else:
                    new_grid[byte_index] &= ~(1 << (7 - bit_index))
        return replace(self, grid=new_grid)

    def clear_lines(self) -> tuple["Board", int]:
        new_grid = self.grid.copy()
        lines_cleared = 0
        for y in range(self.total_height):
            if all(self._get_bit(x, y) for x in range(self.width)):
                for above_y in range(y, self.total_height - 1):
                    for x in range(self.width):
                        self._set_bit(new_grid, x, above_y, self._get_bit(x, above_y + 1))
                for x in range(self.width):
                    self._set_bit(new_grid, x, self.total_height - 1, False)
                lines_cleared += 1
        return replace(self, grid=new_grid), lines_cleared

    def _get_bit(self, x: int, y: int) -> bool:
        byte_index = (x * (self.total_height) + y) // 8
        bit_index = (x * (self.total_height) + y) % 8
        return bool((self.grid[byte_index] & (1 << (7 - bit_index))).any())

    def _set_bit(self, grid: np.ndarray, x: int, y: int, value: bool) -> None:
        byte_index = (x * (self.total_height) + y) // 8
        bit_index = (x * (self.total_height) + y) % 8
        if value:
            grid[byte_index] |= 1 << (7 - bit_index)
        else:
            grid[byte_index] &= ~(1 << (7 - bit_index))

    def collision(self, piece: Piece) -> bool:
        for x_offset, y_offset in piece.shape:
            x = piece.position[0] + x_offset
            y = piece.position[1] + y_offset
            in_bounds = (0 <= x < self.width) and (0 <= y < (self.total_height))
            if not in_bounds:
                return True  # Out of bounds
            if self._get_bit(x, y):
                return True  # Cell is already occupied
        return False

    def count_holes(self):
        holes = 0
        for x in range(self.width):
            found_block = False
            for y in range(self.total_height):
                if self._get_bit(x, y):
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def get_column_heights(self):
        heights = np.zeros(self.width, dtype=int)
        for x in range(self.width):
            for y in range(self.total_height):
                if self._get_bit(x, y):
                    heights[x] = self.total_height - y
                    break
        return heights

    def set_piece_spawn_position(self, piece: Piece) -> Piece:
        # TODO would this make more sense being a method for Piece and not Board????
        return replace(piece, position=self.get_spawn_position(piece))

    def get_spawn_position(self, piece: Piece) -> tuple[np.uint8, np.uint8]:
        """
        Calculate the spawn position of a piece on the board.

        The spawn position is defined as the position that centers the piece horizontally
        and places the top of the piece at the top of the buffer.

        Args:
            piece: The piece to calculate the spawn position for.

        Returns:
            A tuple (x, y) indicating the spawn position of the piece.
        """
        x_values = [i for i, j in piece.shape]
        min_x, max_x = min(x_values), max(x_values)
        piece_width = max_x - min_x + 1
        spawn_x = np.uint8((self.width - piece_width) // 2 - min_x)

        return (spawn_x, self.spawn_y)

    def calculate_well_sums(self) -> np.ndarray:
        heights = self.get_column_heights()
        wells = np.zeros_like(heights, dtype=np.uint8)
        for i in range(len(heights)):
            left = heights[i - 1] if i > 0 else heights[i]
            right = heights[i + 1] if i < len(heights) - 1 else heights[i]
            if heights[i] < left and heights[i] < right:
                wells[i] = min(left, right) - heights[i]
        return wells

    def get_rgb_board(self) -> np.ndarray:
        unpacked_grid = self.get_unpacked_binary_board()
        rgb_grid = np.stack(3 * [unpacked_grid * 255], axis=-1)

        return rgb_grid

    def get_unpacked_binary_board(self) -> np.ndarray:
        unpacked_grid = np.unpackbits(self.grid, axis=0).reshape(self.width, self.total_height)

        return unpacked_grid

    def get_placed_blocks(self):
        """
        Returns a list of placed blocks on the board.

        Each block is represented as a tuple (x, y, color) where (x, y) is the position of the block and color is a tuple (r, g, b) of the color of the block.

        Returns:
            list[tuple[int, int, tuple[int, int, int]]]: A list of placed blocks on the board.
        """
        blocks = []

        for x in range(self.width):
            for y in range(self.total_height):
                if self._get_bit(x, y):
                    blocks.append((x, y, (255, 255, 255)))
        return blocks

    def __str__(self):
        char_map = {True: "■", False: " "}
        grid = [[char_map[self._get_bit(x, y)] for x in range(self.width)] for y in range(self.total_height)]

        board_str = "\nBoard:\n"
        for row in grid:
            board_str += "|" + " ".join(row) + "|\n"
        board_str += "+" + " -" * (self.width - 1) + " +"

        info = f"Width: {self.width}, Height: {self.height}, Buffer Height: {self.buffer_height}\n"
        info += f"Total Height: {self.total_height}, Spawn Y: {self.spawn_y}\n"

        return info + board_str
