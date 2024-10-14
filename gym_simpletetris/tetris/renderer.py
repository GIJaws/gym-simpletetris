import pygame
import numpy as np
from typing import Literal

from gym_simpletetris.tetris.tetris_engine import GameState
from gym_simpletetris.tetris.board import Board
from gym_simpletetris.tetris.pieces import Piece


class Renderer:
    def __init__(
        self,
        width: int,
        height: int,
        buffer_height: int,
        visible_height: int,
        render_mode: Literal["rgb_array", "human"],
        render_fps: int,
        window_size: int,
    ):
        self.width = width
        self.height = height
        self.buffer_height = buffer_height
        self.total_height = self.height + self.buffer_height
        self.visible_height = visible_height or self.height

        if self.visible_height > self.total_height:
            raise ValueError(f"{self.visible_height=} is greater than {self.total_height=}")

        self.render_mode = render_mode
        self.render_fps = render_fps
        self.window_size = window_size
        self.block_size = self.window_size // max(self.width, self.visible_height)

        self.window = None
        self.clock = None
        self.font = None

        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

    def render(self, game_state: GameState):
        if self.render_mode == "rgb_array":
            return self.render_rgb_array(game_state)
        elif self.render_mode == "human":
            return self.render_human(game_state)
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def render_rgb_array(self, board):
        return self._convert_grayscale(board, 640)

    def _convert_grayscale_rgb(self, array):
        shape = array.shape
        shape = (shape[0], shape[1])
        grayscale = np.reshape(array, newshape=(*shape, 1))

        return np.repeat(grayscale, 3, axis=2)

    def render_human(self, game_state: GameState):
        # assume pygame has been intialised when the Renderer object is created

        self.window.fill((0, 0, 0))  # Clear screen with black background

        # Render the game board
        self._render_board(game_state.board.grid)

        # Render ghost piece
        ghost_piece = game_state.get_ghost_piece()
        self._render_piece(game_state.current_piece.shape, ghost_piece.position, ghost_piece.color, ghost=True)

        # Render current piece
        self._render_piece(
            game_state.current_piece.shape, game_state.current_piece.position, game_state.current_piece.color
        )

        # Render UI components
        self._render_ui(game_state)

        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.render_fps)

        return None  # human render mode should return None

    def _render_board(self, board):
        for y in range(self.total_height - self.visible_height, self.total_height):
            for x in range(self.width):
                color = tuple(board[y][x])

                self._render_piece(((0, 0),), (x, y), color, ghost=False)

    def _render_piece(self, shape, anchor, color, ghost=False):
        for i, j in shape:
            x, y = int(anchor[0] + i), int(anchor[1] + j)
            if 0 <= x < self.width and 0 <= y < self.total_height:
                y_screen = (y - self.total_height + self.visible_height) * self.block_size
                rect = pygame.Rect(
                    x * self.block_size,
                    y_screen,
                    self.block_size,
                    self.block_size,
                )
                if ghost:
                    pygame.draw.rect(self.window, color, rect, 1)  # Draw only outline for ghost piece
                else:
                    if any(color):  # If the color is not black
                        pygame.draw.rect(self.window, color, rect)
                    pygame.draw.rect(self.window, (50, 50, 50), rect, 1)  # Grid lines

    def _render_ui(self, game_state: GameState):
        x_offset = 2 * ((self.window_size - self.width * self.block_size) // 2)
        y_offset = 10
        gap = 30

        # Render available information from GameState
        for key, value in [
            ("Score", game_state.score),
            ("Level", game_state.level),
            ("FPS", round(self.clock.get_fps(), 2)),
        ]:
            self._render_text(f"{key}: {value}", (x_offset, y_offset))
            y_offset += gap

        # Commented out unavailable information
        """
        # TODO: Implement these in GameState
        # ("is_current_finesse", game_state.is_current_finesse),
        # ("is_finesse_complete", game_state.is_finesse_complete),
        # ("Lines", game_state.total_lines_cleared),
        """

        # TODO: Implement action handling in GameState
        """
        actions_str = "+".join([BASIC_ACTIONS.get(act, "NO_ACTION") for act in game_state.actions])
        self._render_text(f"{actions_str}", (x_offset + 80, y_offset))
        y_offset += gap
        """

        # TODO need to fix the render_piece_preview to work with multiple pieces and use Pieces
        # Render held piece if available
        if game_state.held_piece:
            self._render_piece_preview(game_state.held_piece, (x_offset + 80, y_offset), "Held")
            y_offset += gap

        # Render next piece
        if game_state.next_pieces:
            self._render_piece_preview(game_state.next_pieces[0], (x_offset, y_offset), "Next")

        # TODO: Implement additional UI elements as GameState evolves

    def _render_text(self, text, pos):
        text_surface = self.font.render(text, True, (255, 255, 255))
        self.window.blit(text_surface, pos)

    def _render_piece_preview(self, piece: Piece, pos: tuple[int, int], label: str):
        preview_size = 80
        block_size = preview_size // 5  # Slightly smaller blocks
        spacing = 10  # Vertical spacing between pieces

        # Draw label
        self._render_text(label, pos)

        # Draw background
        total_height = preview_size * len(piece.shape) + spacing * (
            len(piece.shape) - 1
        )  # TODO verify total_height is implemented and working correctly
        pygame.draw.rect(self.window, (50, 50, 50), (pos[0], pos[1], preview_size, total_height))

        # Calculate bounds of the shape
        min_x = min(x for x, y in piece.shape)
        max_x = max(x for x, y in piece.shape)
        min_y = min(y for x, y in piece.shape)
        max_y = max(y for x, y in piece.shape)
        width = max_x - min_x + 1
        height = max_y - min_y + 1

        # Center the piece
        offset_x = (preview_size - width * block_size) // 2
        offset_y = (preview_size - height * block_size) // 2

        for x, y in piece.shape:
            pygame.draw.rect(
                self.window,
                piece.color,
                (
                    pos[0] + offset_x + (x - min_x) * block_size,
                    pos[1] + offset_y + (y - min_y) * block_size,
                    block_size - 1,
                    block_size - 1,
                ),
            )

    def _convert_grayscale(self, board, size):
        # breakpoint()
        arr = np.array(board, dtype=np.uint8)

        shape = arr.shape[:2]

        # Calculate block size based on the larger dimension to ensure everything fits
        block_size = min(size // shape[0], size // shape[1])
        gap_size = max(1, block_size // 10)  # Adjust gap size relative to block size

        # Calculate total width and height of the game area
        inner_width = (block_size + gap_size) * shape[0] + gap_size
        inner_height = (block_size + gap_size) * shape[1] + gap_size

        # Calculate padding to center the board
        padding_width = size - inner_width
        padding_height = size - inner_height

        # Create result array with dark background
        result = np.full((size, size, 3), 30, dtype=np.uint8)

        for i in range(shape[0]):
            for j in range(shape[1]):
                x_start = padding_width + (i * (block_size + gap_size))
                y_start = padding_height + (j * (block_size + gap_size))
                if np.any(arr[i, j]):
                    # Filled blocks
                    result[y_start : y_start + block_size, x_start : x_start + block_size] = arr[i, j]
                else:
                    # Empty blocks - set to a slightly lighter gray
                    result[y_start : y_start + block_size, x_start : x_start + block_size] = 50

        return result

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()  # TODO SHOULD I DO THE SELF = NONE BELOW?? VVVVVVVVVVVVVVVVVVVVVVVVV
            self.window = None
            self.clock = None
            self.font = None
