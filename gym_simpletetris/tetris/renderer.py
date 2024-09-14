import numpy as np
import pygame


class Renderer:
    def __init__(self, width, height, window_size=512):
        self.width = width
        self.height = height
        self.window_size = window_size
        self.window = None
        self.clock = None

    def render_rgb_array(self, board):
        obs = self._convert_grayscale(board, 160)
        return self._convert_grayscale_rgb(obs)

    def render_human(self, board):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        obs = np.transpose(board)
        obs = self._convert_grayscale(obs, self.window_size)
        obs = self._convert_grayscale_rgb(obs)

        pygame.pixelcopy.array_to_surface(self.window, obs)
        canvas = pygame.surfarray.make_surface(obs)
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(30)  # Assuming 30 FPS, adjust as needed
        return obs

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

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
