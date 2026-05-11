import numpy as np
from numpy.typing import NDArray
from gym_simpletetris.core.pieces import Piece
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Board:
    buffer_height: int
    width: int
    height: int
    # ? Assumption that any changes must be made on a copy of the grid and not done inplace
    grid: NDArray[np.uint8]  # A 2D grid where each cell is either 0 (False) or 1 (True)
    rgb_grid: NDArray[np.uint8]  # A 3D grid (height, width, 3) for RGB colors
    spawn_y: int = field(init=False)
    total_height: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "spawn_y", self.buffer_height - 1)
        object.__setattr__(self, "total_height", self.height + self.buffer_height)

    @staticmethod
    def create_board(width: int, height: int, buffer_height: int) -> "Board":

        grid = np.zeros((height + buffer_height, width), dtype=np.uint8)  # Create a 2D grid initialized to False (0)
        rgb_grid = np.zeros((height + buffer_height, width, 3), dtype=np.uint8)
        return Board(buffer_height=buffer_height, width=width, height=height, grid=grid, rgb_grid=rgb_grid)

    def place_piece(self, piece: Piece, block: bool | np.uint8 = True) -> "Board":
        block = np.uint8(block)
        new_grid = self.grid.copy()
        new_rgb_grid = self.rgb_grid.copy()
        for x_offset, y_offset in piece.shape:
            x = piece.position[0] + x_offset
            y = piece.position[1] + y_offset
            if (0 <= x < self.width) and (0 <= y < self.total_height):
                new_grid[y, x] = block
                new_rgb_grid[y, x] = piece.color
        return replace(self, grid=new_grid, rgb_grid=new_rgb_grid)

    def clear_lines(self) -> tuple["Board", int]:
        new_grid = self.grid.copy()
        new_rgb_grid = self.rgb_grid.copy()
        lines_cleared = 0

        for y in range(self.total_height):
            if new_grid[y, :].all():  # If the entire row is filled
                new_grid[1 : y + 1, :] = new_grid[:y, :]  # Shift everything above the cleared line down
                new_grid[0, :] = 0  # Clear the top line
                new_rgb_grid[1 : y + 1, :] = new_rgb_grid[:y, :]  # Shift RGB values as well
                new_rgb_grid[0, :] = [0, 0, 0]  # Clear the top line in RGB grid
                lines_cleared += 1

        return replace(self, grid=new_grid, rgb_grid=new_rgb_grid), lines_cleared

    def collision(self, piece: Piece) -> bool:
        x_offsets = piece.shape[:, 0]
        y_offsets = piece.shape[:, 1]
        x_positions = piece.position[0] + x_offsets
        y_positions = piece.position[1] + y_offsets

        # Convert positions to integers for indexing
        x_positions = x_positions.astype(int)
        y_positions = y_positions.astype(int)

        # Check if any positions are out of bounds
        x_in_bounds = (0 <= x_positions) & (x_positions < self.width)
        y_in_bounds = (0 <= y_positions) & (y_positions < self.total_height)
        in_bounds = x_in_bounds & y_in_bounds

        if not np.all(in_bounds):
            return True  # Out of bounds

        # Check if any in-bounds positions are occupied
        occupied = self.grid[y_positions, x_positions]
        if np.any(occupied):
            return True  # Cell is already occupied

        return False

    def count_holes(self) -> int:
        holes = 0
        for x in range(self.width):
            found_block = False
            for y in range(self.total_height):
                if self.grid[y, x]:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def get_column_heights(self) -> np.ndarray:
        heights = np.zeros(self.width, dtype=int)
        for x in range(self.width):
            for y in range(self.total_height):
                if self.grid[y, x]:
                    heights[x] = self.total_height - y
                    break
        return heights

    def set_piece_spawn_position(self, piece: Piece) -> Piece:
        return replace(piece, position=self.get_spawn_position(piece))

    def get_spawn_position(self, piece: Piece) -> tuple[int, int]:
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
        spawn_x = int((self.width - piece_width) // 2 - min_x)

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

    def calculate_total_well_sum(self) -> int:
        wells = self.calculate_well_sums()
        return int(np.sum(wells))

    def calculate_bumpiness(self) -> int:
        """
        Calculate the bumpiness of the board.

        The bumpiness is defined as the sum of the absolute differences between adjacent columns.

        Returns:
            int: The bumpiness of the board.
        """
        heights = self.get_column_heights()
        bumpiness = 0
        for i in range(1, len(heights)):
            bumpiness += abs(heights[i] - heights[i - 1])
        return bumpiness

    def get_rgb_board(self) -> np.ndarray:
        rgb_grid = np.stack([self.grid * 255] * 3, axis=-1)
        return rgb_grid

    def get_placed_blocks(self) -> list[tuple[int, int, tuple[int, int, int]]]:
        """
        Returns a list of placed blocks on the board.

        Each block is represented as a tuple (x, y, color) where (x, y) is the position of the block and color is a tuple (r, g, b) of the color of the block.

        Returns:
            list[tuple[int, int, tuple[int, int, int]]]: A list of placed blocks on the board.
        """
        blocks: list[tuple[int, int, tuple[int, int, int]]] = []
        for y in range(self.total_height):
            for x in range(self.width):
                if self.grid[y, x]:
                    blocks.append((x, y, tuple(self.rgb_grid[y, x])))
        return blocks

    def __str__(self):
        char_map = {1: "■", 0: " "}
        board_str = "\nBoard:\n"
        for y in range(self.total_height):  # Start from top row
            row = [char_map[self.grid[y, x]] for x in range(self.width)]
            board_str += "|" + " ".join(row) + "|\n"
        board_str += "+" + " -" * (self.width - 1) + " +"

        info = f"Width: {self.width}, Height: {self.height}, Buffer Height: {self.buffer_height}\n"
        info += f"Total Height: {self.total_height}, Spawn Y: {self.spawn_y}\n"

        return info + board_str
