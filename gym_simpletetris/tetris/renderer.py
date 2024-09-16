import pygame
import numpy as np


class Renderer:
    def __init__(self, width, height, window_size=512):
        self.width = width
        self.height = height
        self.window_size = window_size
        self.window = None
        self.clock = None
        self.font = None

    def render_rgb_array(self, board):
        obs = self._convert_grayscale(board, 160)
        return self._convert_grayscale_rgb(obs)

    def render_human(self, board, game_state):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.font is None:
            self.font = pygame.font.Font(None, 24)

        self.window.fill((0, 0, 0))  # Clear screen

        # Render the game board
        self._render_board(board)

        # Render UI components
        self._render_ui(game_state)

        pygame.display.update()
        self.clock.tick(30)  # 30 FPS

    def _render_board(self, board):
        block_size = self.window_size // max(self.width, self.height)
        for y in range(self.height):
            for x in range(self.width):
                if board[x][y]:
                    pygame.draw.rect(
                        self.window,
                        (255, 255, 255),
                        (x * block_size, y * block_size, block_size, block_size),
                    )
                pygame.draw.rect(
                    self.window,
                    (50, 50, 50),
                    (x * block_size, y * block_size, block_size, block_size),
                    1,
                )

    def _render_ui(self, game_state):
        score_text = self.font.render(
            f"Score: {game_state['score']}", True, (255, 255, 255)
        )
        self.window.blit(score_text, (10, 10))

        lines_text = self.font.render(
            f"Lines: {game_state['lines_cleared']}", True, (255, 255, 255)
        )
        self.window.blit(lines_text, (10, 40))

        level_text = self.font.render(
            f"Level: {game_state['level']}", True, (255, 255, 255)
        )
        self.window.blit(level_text, (10, 70))

        # Render held piece
        self._render_piece(
            game_state["held_piece"], (self.window_size - 100, 10), (100, 100)
        )

        # Render next piece
        self._render_piece(
            game_state["next_piece"], (self.window_size - 100, 120), (100, 100)
        )

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
            np.repeat(
                [0, len(arr)], [padding_width, size - (padding_width + len(arr))]
            ),
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
            pygame.quit()
            self.window = None
            self.clock = None
            self.font = None
