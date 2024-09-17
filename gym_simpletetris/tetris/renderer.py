import pygame
import numpy as np

from gym_simpletetris.tetris.tetris_shapes import SHAPES


class Renderer:
    def __init__(self, width, height, render_mode, render_fps, window_size=512):
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.window_size = window_size
        self.block_size = self.window_size // max(self.width, self.height)
        self.window = None
        self.clock = None
        self.font = None

        # Initialize Pygame here only if human playing
        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

    def render(self, board, gamestate):
        if self.render_mode == "rgb_array":
            return self.render_rgb_array(board)
        elif self.render_mode == "human":
            return self.render_human(board, gamestate)

    def render_rgb_array(self, board):
        obs = self._convert_grayscale(board, 160)
        return self._convert_grayscale_rgb(obs)

    def render_human(self, board, game_state):
        # assume pygame has been intialised when the Renderer object is created

        self.window.fill((0, 0, 0))  # Clear screen with black background

        # Render the game board
        self._render_board(board)

        # Render UI components
        self._render_ui(game_state)

        pygame.event.pump()

        pygame.display.update()

        self.clock.tick(self.render_fps)

        print(self.clock.get_fps())
        return None

    def _render_board(self, board):
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                if board[x][y]:
                    pygame.draw.rect(self.window, (255, 255, 255), rect)  # White for filled blocks
                pygame.draw.rect(self.window, (50, 50, 50), rect, 1)  # Grid lines

    def _render_ui(self, game_state):
        # Render score, lines, and level
        self._render_text(f"Score: {game_state['score']}", (10, 10))
        self._render_text(f"Lines: {game_state['lines_cleared']}", (10, 40))
        self._render_text(f"Level: {game_state['level']}", (10, 70))

        # Render held piece
        self._render_piece_preview(game_state["held_piece"], (self.window_size - 100, 10), "Held")

        # Render next piece
        self._render_piece_preview(game_state["next_piece"], (self.window_size - 100, 120), "Next")

    def _render_text(self, text, pos):
        text_surface = self.font.render(text, True, (255, 255, 255))
        self.window.blit(text_surface, pos)

    def _render_piece(self, piece, pos, size):
        if piece:
            block_size = min(size[0], size[1]) // 4
            for x, y in piece:
                pygame.draw.rect(
                    self.window,
                    (200, 200, 200),
                    (
                        pos[0] + (x + 1) * block_size,
                        pos[1] + (y + 1) * block_size,
                        block_size,
                        block_size,
                    ),
                )

    def _render_piece_preview(self, pieces, pos, label):
        if pieces:
            preview_size = 80
            block_size = preview_size // 4

            # Draw label
            self._render_text(label, (pos[0], pos[1] - 30))

            # Draw background
            pygame.draw.rect(self.window, (50, 50, 50), (pos[0], pos[1], preview_size, preview_size * len(pieces)))

            # Draw pieces
            for i, piece in enumerate(pieces):
                shape = SHAPES[piece]
                for x, y in shape:
                    pygame.draw.rect(
                        self.window,
                        (255, 255, 255),
                        (
                            pos[0] + (x + 1) * block_size,
                            pos[1] + (y + 1) * block_size + i * preview_size,
                            block_size - 1,
                            block_size - 1,
                        ),
                    )

    def _convert_grayscale(self, board, size):
        border_shade = 0
        background_shade = 128
        piece_shade = 190

        arr = np.array(board, dtype=np.uint8)
        arr = np.transpose(arr)

        shape = arr.shape
        limiting_dim = max(shape[0], shape[1])

        gap_size = (size // 100) + 1
        block_size = ((size - (2 * gap_size)) // limiting_dim) - gap_size

        inner_width = gap_size + (block_size + gap_size) * shape[0]
        inner_height = gap_size + (block_size + gap_size) * shape[1]

        padding_width = (size - inner_width) // 2
        padding_height = (size - inner_height) // 2

        arr[arr == 0] = background_shade
        arr[arr == 1] = piece_shade

        arr = np.repeat(arr, block_size, axis=0)
        arr = np.repeat(arr, block_size, axis=1)

        arr = np.insert(
            arr,
            np.repeat(
                [block_size * x for x in range(shape[0] + 1)],
                [gap_size for _ in range(shape[0] + 1)],
            ),
            background_shade,
            axis=0,
        )
        arr = np.insert(
            arr,
            np.repeat(
                [block_size * x for x in range(shape[1] + 1)],
                [gap_size for _ in range(shape[1] + 1)],
            ),
            background_shade,
            axis=1,
        )

        arr = np.insert(
            arr,
            np.repeat([0, len(arr)], [padding_width, size - (padding_width + len(arr))]),
            border_shade,
            axis=0,
        )
        arr = np.insert(
            arr,
            np.repeat(
                [0, len(arr[0])],
                [padding_height, size - (padding_height + len(arr[0]))],
            ),
            border_shade,
            axis=1,
        )

        return arr

    def _convert_grayscale_rgb(self, array):
        shape = array.shape
        shape = (shape[0], shape[1])
        grayscale = np.reshape(array, newshape=(*shape, 1))

        return np.repeat(grayscale, 3, axis=2)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()  # TODO SHOULD I DO THE SELF = NONE BELOW?? VVVVVVVVVVVVVVVVVVVVVVVVV
            self.window = None
            self.clock = None
            self.font = None
