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

    def convert_board_data_OLD(self, board_data):
        if self.obs_type == "binary":
            return np.array(board_data != 0, dtype=np.uint8)
        elif self.obs_type == "grayscale":
            return np.array(board_data, dtype=np.uint8)
        elif self.obs_type == "rgb":
            rgb_board = np.zeros((board_data.shape[0], board_data.shape[1], 3), dtype=np.uint8)
            for y in range(board_data.shape[0]):
                for x in range(board_data.shape[1]):
                    if board_data[y, x] != 0:
                        rgb_board[y, x] = board_data[y, x]
            return rgb_board
        else:
            raise ValueError(f"Unsupported observation type: {self.obs_type}")

    @staticmethod
    def convert_board_data(board_data, input_type, output_type):
        if input_type == output_type:
            return board_data

        if output_type == "binary":
            return np.array(board_data != 0, dtype=np.uint8)
        elif output_type == "grayscale":
            if input_type == "binary":
                return np.array(board_data * 255, dtype=np.uint8)
            else:  # input_type is "rgb"
                return np.array(np.mean(board_data, axis=2), dtype=np.uint8)
        elif output_type == "rgb":
            if input_type == "binary":
                return np.repeat(np.array(board_data * 255, dtype=np.uint8)[:, :, np.newaxis], 3, axis=2)
            elif input_type == "grayscale":
                return np.repeat(board_data[:, :, np.newaxis], 3, axis=2)
        else:
            raise ValueError(f"Unsupported output type: {output_type}")

        return board_data

    def format_blocks(self, game_state):
        # Process board into an rgb array for rendering the board based on obs_type
        grid = self.convert_board_data(game_state.board.grid, self.obs_type, "rgb")
        blocks = []

        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if np.any(grid[y, x]):
                    color = tuple(grid[y, x])
                    blocks.append((x, y, color))

        return blocks

    @staticmethod
    def render_piece_data(piece):
        # Process piece data into a format suitable for rendering
        shape = piece.shape
        position = piece.position
        color = piece.color
        blocks = []

        for i, j in np.argwhere(shape):
            x = position[0] + j
            y = position[1] + i
            blocks.append((x, y, color))

        return blocks

    @staticmethod
    def convert_board_to_rgb_array(board, visible_height):
        # Convert the board to an RGB array for 'rgb_array' mode
        visible_board = board.grid[-visible_height:, :]
        return visible_board.astype(np.uint8)
