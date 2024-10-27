import warnings
import numpy as np
from gym_simpletetris.render.base_renderer import BaseRenderer
from gym_simpletetris.core.tetris_engine import GameState


class ArrayRenderer(BaseRenderer):
    def __init__(self, width, height, obs_type, visible_height=None, **kwargs):
        warnings.warn(
            "ArrayRenderer is not finished, DO NOT USE, or do idc",
            RuntimeWarning,
        )
        super().__init__(width, height, obs_type, **kwargs)
        self.visible_height = visible_height or height

    def render(self, game_state: GameState) -> np.ndarray:
        """Renders the game state as a 2D array."""

        board = game_state.board.place_piece(game_state.current_piece)
        grid = board.grid if self.obs_type == "binary" else board.rgb_grid
        return grid[:, -self.visible_height :]

    def close(self):
        pass
