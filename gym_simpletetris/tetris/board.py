import numpy as np
from gym_simpletetris.tetris.pieces import Piece
from dataclasses import dataclass, replace, field


@dataclass(frozen=True)
class Board:
    width: int
    height: int
    buffer_height: int
    grid: np.ndarray  # Immutable 2D or 3D grid
    is_colour: bool = field(init=False)

    def __post_init__(self):
        if self.grid.ndim not in (2, 3):
            raise ValueError("Invalid board shape. Expected 2D or 3D array.")
        object.__setattr__(self, "is_colour", self.grid.ndim == 3)

    def place_piece(self, piece: Piece) -> "Board":
        board = self
        new_grid = board.grid.copy()
        color = piece.color if board.is_colour else 1
        for x_offset, y_offset in np.ndindex(piece.shape.shape):
            if piece.shape[x_offset, y_offset] == 1:
                x = piece.position[0] + x_offset
                y = piece.position[1] + y_offset
                if 0 <= x < board.width and 0 <= y < board.height:
                    new_grid[y, x] = color
        # TODO Should I create new dataclass or use replace?????
        return Board(width=board.width, height=board.height, grid=new_grid, buffer_height=board.buffer_height)

    def clear_lines(self) -> tuple["Board", int]:
        """
        Clear any completed lines and return the new board and the number of lines cleared.

        Returns a tuple (new_board, lines_cleared) where new_board is the updated board state and lines_cleared is the
        number of lines that were cleared.
        """
        board = self
        new_grid = board.grid.copy()

        # Find all the full lines in the grid
        full_lines = [row for row in new_grid if np.all(row != 0)]

        # Create a new grid by removing the full lines and adding new empty lines at the top
        new_grid_list = [row for row in new_grid if not np.all(row != 0)]
        lines_cleared = len(full_lines)
        for _ in range(lines_cleared):
            new_grid_list.insert(0, np.zeros(board.width, dtype=int))
        new_grid = np.array(new_grid_list)

        # Return the new board and the number of lines cleared
        return (
            # TODO Should I create new dataclass or use replace?????
            Board(width=board.width, height=board.height, grid=new_grid, buffer_height=board.buffer_height),
            lines_cleared,
        )

    def collision(self, piece: Piece) -> bool:
        """
        Check if the piece collides with the board boundaries or existing blocks.
        """
        for x_offset, y_offset in np.argwhere(piece.shape):
            x = piece.position[0] + x_offset
            y = piece.position[1] + y_offset
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                return True  # Out of bounds
            if self.grid[y, x]:
                return True  # Cell is already occupied
        return False

    @staticmethod
    def simplify_board(board: np.ndarray) -> np.ndarray:
        if board.ndim == 3:
            return np.any(board != 0, axis=2).astype(np.float32)  # TODO should this be float32??????
        elif board.ndim == 2:
            return board.astype(np.float32)  # TODO should this be float32??????
        else:
            raise ValueError("Invalid board shape. Expected 2D or 3D array.")

    def count_holes(self):
        """Count the number of holes in the board."""
        board = Board.simplify_board(self.grid) if self.is_colour else self.grid
        holes = 0

        num_cols, num_rows = board.shape
        for col in range(num_cols):
            col_data = board[col, :]
            filled_indices = np.where(col_data != 0)[0]
            if len(filled_indices) == 0:
                continue  # No filled cells in this column
            highest_filled_row = filled_indices[0]
            holes += np.sum(col_data[highest_filled_row + 1 :] == 0)
        return holes

    def get_column_heights(self):
        non_zero_mask = self.grid != 0

        heights = self.grid.shape[1] - np.argmax(non_zero_mask, axis=1)
        return np.where(non_zero_mask.any(axis=1), heights, 0)

    def set_piece_spawn_position(self, piece: Piece) -> Piece:
        # TODO would this make more sense being a method for Piece and not Board????
        return replace(piece, position=self.get_spawn_position(piece))

    def get_spawn_position(self, piece: Piece) -> tuple[int, int]:
        x_values = [i for i, j in piece.shape]
        min_x, max_x = min(x_values), max(x_values)
        piece_width = max_x - min_x + 1
        spawn_x = (self.width - piece_width) // 2 - min_x
        spawn_y = self.buffer_height - 1
        return (spawn_x, spawn_y)

    def calculate_well_sums(self) -> np.ndarray:
        """
        Calculate the well sums of the board.
        """
        heights = self.get_column_heights()
        wells = np.zeros_like(heights)
        for i in range(len(heights)):
            left = heights[i - 1] if i > 0 else heights[i]
            right = heights[i + 1] if i < len(heights) - 1 else heights[i]
            if heights[i].any() < left and heights[i].any() < right:
                wells[i] = min(left, right) - heights[i]
        return wells
