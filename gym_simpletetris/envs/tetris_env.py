import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .game_state import GameState
from .renderer import TetrisRenderer
from .input_handler import RandomInputHandler


class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, width=10, height=20, render_mode="rgb_array"):
        self.game_state = GameState(width, height)
        self.action_space = spaces.Discrete(4)  # left, right, rotate, drop
        self.observation_space = spaces.Box(
            0, 1, shape=(width, height), dtype=np.float32
        )
        self.render_mode = render_mode
        self.renderer = TetrisRenderer(width, height)
        self.input_handler = RandomInputHandler(self.action_space)

    def step(self, action):
        # Use the action provided by the environment
        if action == 0:
            self.game_state.move_left()
        elif action == 1:
            self.game_state.move_right()
        elif action == 2:
            self.game_state.rotate()
        elif action == 3:
            self.game_state.drop()

        self.game_state.lock_piece()

        reward = self.game_state.score
        done = self.game_state.is_game_over()
        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game_state = GameState(
            self.game_state.board.width, self.game_state.board.height
        )
        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def render(self):
        return self.renderer.render(self.game_state, mode=self.render_mode)

    def close(self):
        self.renderer.close()

    def _get_observation(self):
        return self.game_state.board.grid

    def _get_info(self):
        return {
            "score": self.game_state.score,
            "lines_cleared": self.game_state.lines_cleared,
        }

    def get_action(self):
        return self.input_handler.get_action(self.game_state)
