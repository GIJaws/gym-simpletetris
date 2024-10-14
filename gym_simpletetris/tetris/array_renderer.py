from gym_simpletetris.tetris.base_renderer import BaseRenderer


class ArrayRenderer(BaseRenderer):
    def __init__(self, width, height, obs_type, visible_height=None, **kwargs):
        super().__init__(width, height, obs_type, **kwargs)
        self.visible_height = visible_height or height

    def render(self, game_state):
        # Convert the game state to an array representation
        board_data = game_state.board.grid[-self.visible_height :]
        return self.convert_board_data(board_data, self.obs_type, "rgb")

    def close(self):
        pass  # No resources to clean up
