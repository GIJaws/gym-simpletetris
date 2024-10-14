from dataclasses import replace
from typing import Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces


from gym_simpletetris.tetris.human_renderer import HumanRenderer
from gym_simpletetris.tetris.array_renderer import ArrayRenderer
from gym_simpletetris.tetris.pieces import PieceQueue
from gym_simpletetris.tetris.tetris_engine import GameState
from gym_simpletetris.tetris.game_actions import GameAction


class TetrisEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "obs_types": ["binary", "grayscale", "rgb"],
        "render_fps": 60,
        "initial_level": 1,
        "num_lives": 1,
        "lock_delay": 0,
        "preview_size": 4,
    }

    def __init__(
        self,
        width,
        height,
        buffer_height,
        visible_height,
        obs_type="binary",
        render_mode="rgb_array",
        window_size=512,
        initial_level=1,
        num_lives=1,
        render_fps=60,
        preview_size=4,
    ):
        self.width = width
        self.height = height
        self.buffer_height = buffer_height
        self.visible_height = visible_height
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.window_size = window_size
        self.render_fps = render_fps

        # Initialize the renderer
        self.renderer = self._create_renderer()

        self.action_space = spaces.Discrete(len(GameAction))
        self.observation_space = self._get_observation_space()

        self.total_steps = 0
        self.piece_queue = PieceQueue(preview_size)
        self.initial_game_state = GameState.create_initial_game_state(
            width=self.width,
            height=self.height,
            buffer_height=self.buffer_height,
            current_piece=self.piece_queue.next_piece(),
            next_pieces=tuple(self.piece_queue.get_preview()),
            initial_level=initial_level,
            held_piece=None,
            is_color=(obs_type == "rgb"),
        )
        self.num_lives = num_lives
        self.deaths = 0

        self.game_state = self.initial_game_state

    def _create_renderer(self):
        if self.render_mode == "human":
            return HumanRenderer(
                width=self.width,
                height=self.height + self.buffer_height,
                block_size=self.window_size // self.visible_height,
                fps=self.render_fps,
                visible_height=self.visible_height,
                obs_type=self.obs_type,
            )
        elif self.render_mode in ["rgb_array", "grayscale"]:
            return ArrayRenderer(
                width=self.width,
                height=self.height + self.buffer_height,
                visible_height=self.visible_height,
                obs_type=self.obs_type,
            )
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def _get_observation_space(self):
        if self.obs_type == "binary":
            shape = (self.visible_height, self.width)
            return spaces.Box(low=0, high=1, shape=shape, dtype=np.uint8)
        elif self.obs_type in ["grayscale", "rgb"]:
            shape = (self.visible_height, self.width, 3)
            return spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported observation type: {self.obs_type}")

    def step(self, action):
        actions = GameAction.from_index(action)
        og_score = self.game_state.score
        self.game_state = self.game_state.step(actions=actions)

        # Check if we need to fetch more pieces
        while len(self.game_state.next_pieces) < self.piece_queue.preview_size:
            new_piece = self.piece_queue.next_piece()
            self.game_state = replace(self.game_state, next_pieces=tuple(self.game_state.next_pieces) + (new_piece,))

        if self.render_mode == "human":
            self.render()

        # Return the standard Gym tuple
        observation = self._get_observation()
        reward = self.game_state.score - og_score  # TODO this reward is a stub
        terminated = self.game_state.game_over
        truncated = False  # TODO define conditions for truncation
        info = self.game_state.info

        return observation, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)

        # Refill the bag with new random pieces
        self.piece_queue.refill_bag()

        self.game_state = self.initial_game_state.create_reset_state(
            self.piece_queue.next_piece(), tuple(self.piece_queue.get_preview())
        )

        if self.render_mode == "human":
            self.render()

        return self._get_observation(), self.game_state.info

    # def _get_observation(self):
    #     board = self.game_state.get_full_board().grid
    #     # Extract only the visible part of the board if necessary
    #     visible_board = board[-self.renderer.visible_height :, :]

    #     # TODO do I need to do this obs logic for simplifying the board? also this isn't concating the non visbile part of the board off?
    #     if self.obs_type == "binary":
    #         return (visible_board != 0).astype(np.uint8)
    #     elif self.obs_type in ["grayscale", "rgb"]:
    #         # TODO do we need to convert to grayscale here? probs using the renderer
    #         return visible_board.astype(np.uint8)
    #     else:
    #         raise ValueError(f"Unsupported observation type: {self.obs_type}")

    def _get_observation(self):
        board = self.game_state.get_full_board().grid
        # Extract only the visible part of the board
        visible_board = board[-self.visible_height :, :]

        if self.obs_type == "binary":
            # TODO probs will need to change this to return  `astype(np.uint8)`
            # TODO use gamestates methods to get bool board
            return (visible_board != 0).astype(np.bool_)
        elif self.obs_type in ["grayscale", "rgb"]:
            return visible_board.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported observation type: {self.obs_type}")

    def render(self):

        return self.renderer.render(self.game_state)

    def close(self):
        self.renderer.close()
