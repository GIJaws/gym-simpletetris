import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from .tetris_engine import TetrisEngine


def convert_grayscale(board, size):
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
            [0, len(arr[0])], [padding_height, size - (padding_height + len(arr[0]))]
        ),
        border_shade,
        axis=1,
    )

    return arr


def convert_grayscale_rgb(array):
    shape = array.shape
    shape = (shape[0], shape[1])
    grayscale = np.reshape(array, newshape=(*shape, 1))

    return np.repeat(grayscale, 3, axis=2)


class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    # TODO: Add more reward options e.g. wells
    # TODO: Reorganise on next major release
    def __init__(
        self,
        width=10,
        height=20,
        obs_type="ram",
        extend_dims=False,
        render_mode="rgb_array",
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
        self.window_size = 512

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
        self.window = None
        self.clock = None

        if obs_type == "ram":
            if extend_dims:
                self.observation_space = spaces.Box(
                    0, 1, shape=(width, height, 1), dtype=np.float32
                )
            else:
                self.observation_space = spaces.Box(
                    0, 1, shape=(width, height), dtype=np.float32
                )
        elif obs_type == "grayscale":
            if extend_dims:
                self.observation_space = spaces.Box(
                    0, 1, shape=(84, 84, 1), dtype=np.float32
                )
            else:
                self.observation_space = spaces.Box(
                    0, 1, shape=(84, 84), dtype=np.float32
                )
        elif obs_type == "rgb":
            self.observation_space = spaces.Box(
                0, 1, shape=(84, 84, 3), dtype=np.float32
            )

    def _get_info(self):
        return self.engine.get_info()

    def step(self, action):
        state, reward, done = self.engine.step(action)
        observation = self._observation(state=state)
        info = self._get_info()

        # In the modern Gym API, we need to return a boolean for truncated
        # For Tetris, we can consider the episode as truncated if it's done
        truncated = done

        return observation, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.engine.clear()
        observation = self._observation(state=state)
        info = self._get_info()
        return observation, info

    def _observation(self, mode=None, state=None, extend_dims=None):
        obs = state

        if obs is None:
            obs = self.engine.render()

        new_mode = self.obs_type if mode is None else mode

        if new_mode == "ram":
            extend = self.extend_dims if extend_dims is None else extend_dims

            return (
                np.reshape(obs, newshape=(self.width, self.height, 1))
                if extend
                else obs
            )
        else:
            obs = convert_grayscale(obs, 84)

            if new_mode == "grayscale":
                extend = self.extend_dims if extend_dims is None else extend_dims

                return np.reshape(obs, newshape=(84, 84, 1)) if extend else obs
            else:
                return convert_grayscale_rgb(obs)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        elif self.render_mode == "human":
            return self._render_human()
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def _render_rgb_array(self):
        obs = self.engine.render()
        obs = convert_grayscale(obs, 160)
        return convert_grayscale_rgb(obs)

    def _render_human(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        obs = self.engine.render()
        obs = np.transpose(obs)
        obs = convert_grayscale(obs, self.window_size)
        obs = convert_grayscale_rgb(obs)

        pygame.pixelcopy.array_to_surface(self.window, obs)
        canvas = pygame.surfarray.make_surface(obs)
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        return obs if self.render_mode == "rgb_array" else None

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
