import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .tetris_engine import TetrisEngine
from .renderer import Renderer


class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        width=10,
        height=20,
        obs_type="ram",
        extend_dims=False,
        render_mode="rgb_array",
        window_size=512,
        reward_step=False,
        penalise_height=False,
        penalise_height_increase=False,
        advanced_clears=False,
        high_scoring=False,
        penalise_holes=False,
        penalise_holes_increase=False,
        lock_delay=0,
        step_reset=False,
        initial_level=1,
    ):
        self.obs_type = obs_type
        self.extend_dims = extend_dims

        self.renderer = Renderer(width, height, render_mode, self.metadata["render_fps"], window_size=window_size)
        self.engine = TetrisEngine(
            width,
            height,
            lock_delay,
            step_reset,
            reward_step,
            penalise_height,
            penalise_height_increase,
            advanced_clears,
            high_scoring,
            penalise_holes,
            penalise_holes_increase,
            initial_level,
        )

        self.action_space = spaces.Discrete(7)
        self.observation_space = self._get_observation_space()

    def _get_observation_space(self):
        if self.obs_type == "ram":
            shape = (
                (self.renderer.width, self.renderer.height, 3)
                if self.extend_dims
                else (self.renderer.width, self.renderer.height, 3)
            )
        elif self.obs_type in ["grayscale", "rgb"]:
            shape = (84, 84, 1) if self.obs_type == "grayscale" and self.extend_dims else (84, 84, 3)
        else:
            raise ValueError(f"Unsupported observation type: {self.obs_type}")

        return spaces.Box(0, 255, shape=shape, dtype=np.uint8)

    def step(self, action):
        state, reward, done = self.engine.step(action)
        if self.renderer.render_mode == "human":
            self.render()
        return self._get_observation(state), reward, done, done, self.engine.get_info()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.engine.clear()
        return self._get_observation(state), self.engine.get_info()

    def _get_observation(self, state):
        if self.obs_type == "ram":
            obs = state
        elif self.obs_type in ["grayscale", "rgb"]:
            obs = self.renderer.render_rgb_array(state)
            if self.obs_type == "grayscale":
                obs = np.mean(obs, axis=-1, keepdims=self.extend_dims)
        else:
            raise ValueError(f"Unsupported observation type: {self.obs_type}")
        return obs.astype(np.uint8)

    def render(self):
        board, shape, ghost_anchor, ghost_color = self.engine.render()
        return self.renderer.render(board, self.engine.get_info(), shape, ghost_anchor, ghost_color)

    def close(self):
        self.renderer.close()
