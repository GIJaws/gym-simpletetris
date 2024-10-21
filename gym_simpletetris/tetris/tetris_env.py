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
        # "obs_types": ["binary", "grayscale", "rgb"],
        "obs_types": ["binary"],
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
        msg = f"Initialising TetrisEnv with width={width}, height={height}, buffer_height={buffer_height}, visible_height={visible_height}, obs_type={obs_type}, render_mode={render_mode}, window_size={window_size}, initial_level={initial_level}, num_lives={num_lives}, render_fps={render_fps}, preview_size={preview_size}"
        print(msg)
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

        # TODO implement total_score and total_lines_cleared
        self.total_score = None
        self.total_lines_cleared = None

        self.num_lives = num_lives
        self.current_lives = self.num_lives
        self.deaths = 0
        self.initial_level = initial_level

        self.piece_queue = PieceQueue(preview_size)
        self.game_state = self._create_initial_game_state()

    def _create_initial_game_state(self):
        self.piece_queue.refill_bag()
        return GameState.create_initial_game_state(
            width=self.width,
            height=self.height,
            buffer_height=self.buffer_height,
            current_piece=self.piece_queue.next_piece(),
            next_pieces=tuple(self.piece_queue.get_preview()),
            initial_level=self.initial_level,
            held_piece=None,
        )

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
        elif self.render_mode in ["rgb_array"]:
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
        elif self.obs_type in ["rgb"]:
            shape = (self.visible_height, self.width, 3)
            return spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported observation type: {self.obs_type}")

    def step(self, action):
        actions = GameAction.from_index(*action)
        self.game_state = self.game_state.step(actions=actions)

        while len(self.game_state.next_pieces) < self.piece_queue.preview_size:
            new_piece = self.piece_queue.next_piece()
            self.game_state = replace(self.game_state, next_pieces=tuple(self.game_state.next_pieces) + (new_piece,))

        if self.render_mode == "human":
            self.render()

        observation = self._get_observation()
        reward = self.game_state.step_score  # TODO this reward is just the current game score for this step
        info = self.game_state.info

        if self.game_state.game_over:
            self.current_lives -= 1

            if self.current_lives > 0:
                self.game_state = self._create_initial_game_state()
                terminated = False
            else:
                # No lives left; end the game
                terminated = True
        else:
            terminated = False
        truncated = False  # TODO define conditions for truncation
        info["lives_remaining"] = self.current_lives

        # Return the standard Gym tuple
        return observation, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        print("Resetting TetrisEnv")
        super().reset(seed=seed, options=options)

        self.current_lives = self.num_lives

        # Refill the bag with new random pieces
        self.game_state = self._create_initial_game_state()

        if self.render_mode == "human":
            self.render()

        return self._get_observation(), self.game_state.info

    def _get_observation(self):
        board = self.game_state.board.place_piece(self.game_state.current_piece).grid
        # Extract only the visible part of the board
        # TODO do I need to do this obs logic for simplifying the board? also this isn't concating the non visbile part of the board off?

        # visible_board = board[-self.visible_height :, :]

        if self.obs_type in ["binary", "rgb"]:
            return board
        else:
            raise ValueError(f"Unsupported observation type: {self.obs_type}")

    def render(self):
        return self.renderer.render(self.game_state)

    def close(self):
        print("Closing TetrisEnv")
        self.renderer.close()
