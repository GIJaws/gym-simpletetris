import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .tetris_engine import TetrisEngine
from .renderer import Renderer


class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

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
    ):
        self.width = width
        self.height = height
        self.obs_type = obs_type
        self.extend_dims = extend_dims
        self.render_mode = render_mode

        self.renderer = Renderer(width, height, window_size)
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
        )

        self.action_space = spaces.Discrete(7)
        self.observation_space = self._get_observation_space()

    def _get_observation_space(self):
        if self.obs_type == "ram":
            shape = (
                (self.width, self.height, 1)
                if self.extend_dims
                else (self.width, self.height)
            )
        elif self.obs_type in ["grayscale", "rgb"]:
            shape = (
                (84, 84, 1)
                if self.obs_type == "grayscale" and self.extend_dims
                else (84, 84, 3)
            )
        else:
            raise ValueError(f"Unsupported observation type: {self.obs_type}")

        return spaces.Box(0, 1, shape=shape, dtype=np.float32)

    def _get_info(self):
        return self.engine.get_info()

    def step(self, action):
        state, reward, done = self.engine.step(action)
        observation = self._get_observation(state)
        info = self._get_info()
        return observation, reward, done, done, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.engine.clear()
        observation = self._get_observation(state)
        info = self._get_info()
        return observation, info

    def _get_observation(self, state):
        if self.obs_type == "ram":
            obs = state if not self.extend_dims else np.expand_dims(state, axis=-1)
        elif self.obs_type in ["grayscale", "rgb"]:
            obs = self.renderer.render_rgb_array(state)
            if self.obs_type == "grayscale":
                obs = np.mean(obs, axis=-1, keepdims=self.extend_dims)
        else:
            raise ValueError(f"Unsupported observation type: {self.obs_type}")
        return obs.astype(np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            return self.renderer.render_rgb_array(self.engine.render())
        elif self.render_mode == "human":
            return self.renderer.render_human(self.engine.render())
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def close(self):
        self.renderer.close()
