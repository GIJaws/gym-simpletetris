# In tetris/rendering/renderer.py

import pygame
import numpy as np


class TetrisRenderer:
    def __init__(self, width, height, cell_size=30):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen_width = width * cell_size
        self.screen_height = height * cell_size
        self.screen = None
        self.clock = None

    def render(self, game_state, mode="human"):
        if mode == "human":
            return self._render_human(game_state)
        elif mode == "rgb_array":
            return self._render_rgb_array(game_state)

    def _render_human(self, game_state):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            self.clock = pygame.time.Clock()

        self.screen.fill((0, 0, 0))  # Fill with black

        # Draw the grid
        for y in range(self.height):
            for x in range(self.width):
                color = (50, 50, 50)  # Empty cell color
                if game_state.board.grid[x, y] == 1:
                    color = (0, 255, 0)  # Placed piece color
                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size - 1,
                        self.cell_size - 1,
                    ),
                )

        # Draw the current piece
        for i, j in game_state.current_shape:
            x, y = game_state.anchor[0] + i, game_state.anchor[1] + j
            if 0 <= x < self.width and 0 <= y < self.height:
                pygame.draw.rect(
                    self.screen,
                    (255, 0, 0),  # Current piece color
                    (
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size - 1,
                        self.cell_size - 1,
                    ),
                )

        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def _render_rgb_array(self, game_state):
        grid = game_state.board.grid.copy()
        for i, j in game_state.current_shape:
            x, y = game_state.anchor[0] + i, game_state.anchor[1] + j
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[x, y] = 2  # Use 2 to represent the current piece
        return grid

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
