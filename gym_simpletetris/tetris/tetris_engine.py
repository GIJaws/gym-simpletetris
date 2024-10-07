import dataclasses
import random
from typing import Literal
import numpy as np
from gym_simpletetris.tetris.finesse_evaluator import FinesseEvaluator
from gym_simpletetris.tetris.scoring_system import AbstractScoringSystem, ScoringSystem
from gym_simpletetris.tetris.pieces import Piece, PieceType, PieceQueue
import math
import heapq

from dataclasses import dataclass

from enum import Enum, auto

from abc import ABC, abstractmethod


from functools import cached_property, total_ordering


@dataclass(frozen=True)
class Board:
    width: int
    height: int
    buffer_height: int
    grid: np.ndarray  # Immutable 2D or 3D grid
    is_colour: bool = dataclasses.field(init=False)

    def __post_init__(self):
        if self.grid.ndim not in (2, 3):
            raise ValueError("Invalid board shape. Expected 2D or 3D array.")
        object.__setattr__(self, "is_colour", self.grid.ndim == 3)

    def place_piece(self, piece: Piece) -> "Board":
        board = self
        new_grid = board.grid.copy()
        color = piece.color if board.is_colour else 1
        for x_offset, y_offset in np.ndindex(piece.shape.shape):
            if piece.shape[x_offset, y_offset] == 1:
                x = piece.position[0] + x_offset
                y = piece.position[1] + y_offset
                if 0 <= x < board.width and 0 <= y < board.height:
                    new_grid[y, x] = color

        return Board(width=board.width, height=board.height, grid=new_grid, buffer_height=board.buffer_height)

    def clear_lines(self) -> tuple["Board", int]:
        board = self
        new_grid = board.grid.copy()
        new_grid_list = [row for row in new_grid if not np.all(row == 1)]
        lines_cleared = board.height - len(new_grid_list)
        for _ in range(lines_cleared):
            new_grid_list.insert(0, np.zeros(board.width, dtype=int))
        new_grid = np.array(new_grid_list)
        return (
            Board(width=board.width, height=board.height, grid=new_grid, buffer_height=board.buffer_height),
            lines_cleared,
        )

    def collision(self, piece: Piece) -> bool:
        """
        Check if the piece collides with the board boundaries or existing blocks.
        """
        for x_offset, y_offset in np.argwhere(piece.shape):
            x = piece.position[0] + x_offset
            y = piece.position[1] + y_offset
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                return True  # Out of bounds
            if self.grid[y, x]:
                return True  # Cell is already occupied
        return False

    @staticmethod
    def simplify_board(board: np.ndarray) -> np.ndarray:
        if board.ndim == 3:
            return np.any(board != 0, axis=2).astype(np.float32)
        elif board.ndim == 2:
            return board.astype(np.float32)
        else:
            raise ValueError("Invalid board shape. Expected 2D or 3D array.")

    def count_holes(self):
        """Count the number of holes in the board."""
        board = Board.simplify_board(self.grid) if self.is_colour else self.grid
        holes = 0

        num_cols, num_rows = board.shape
        for col in range(num_cols):
            col_data = board[col, :]
            filled_indices = np.where(col_data != 0)[0]
            if len(filled_indices) == 0:
                continue  # No filled cells in this column
            highest_filled_row = filled_indices[0]
            holes += np.sum(col_data[highest_filled_row + 1 :] == 0)
        return holes

    def get_column_heights(self):
        non_zero_mask = self.grid != 0

        heights = self.grid.shape[1] - np.argmax(non_zero_mask, axis=1)
        return np.where(non_zero_mask.any(axis=1), heights, 0)


@dataclass(frozen=True)
class GameState:
    board: Board  # Assumption that the board stored in GameState does not include the current piece that is falling
    current_piece: Piece
    next_pieces: tuple[Piece, ...]
    held_piece: Piece | None
    score: int
    lines_cleared: int
    level: int
    game_over: bool
    hold_used: bool = False
    lock_delay_counter: int = 0
    MAX_LOCK_DELAY: int = 30  # TODO IDK WHAT TO PUT HERE
    current_time: int = 0

    # TODO have list that keeps track of actions (list[list[int]]), the score for each time step, and the number of lines cleared for each time step

    def step(self, actions: list["Action"]) -> "GameState":
        state = self.apply_actions(actions)

        if not state.game_over:
            state = state.apply_gravity_handle_piece_locking()
            state = state.update_game_state()
            state.calculate_score()

        return dataclasses.replace(state, current_time=state.current_time + 1)

    def apply_actions(self, actions: list["Action"]) -> "GameState":
        state = self
        for action in sorted(actions):
            if state.game_over:
                break
            state = action.apply(state)
        return state

    def apply_gravity_handle_piece_locking(self) -> "GameState":
        state = self
        piece_after_gravity = Action.move(state.current_piece, dx=0, dy=1)

        if state.board.collision(piece_after_gravity):
            lock_delay_counter = state.lock_delay_counter + 1
        else:
            lock_delay_counter = 0

        state = dataclasses.replace(state, current_piece=piece_after_gravity, lock_delay_counter=lock_delay_counter)

        if state.lock_delay_counter < state.MAX_LOCK_DELAY:
            return state

        board_after_clear, lines_cleared = state.board.place_piece(state.current_piece).clear_lines()
        return dataclasses.replace(state, board=board_after_clear, lines_cleared=lines_cleared)

    def update_game_state(self) -> "GameState":
        state = self
        next_piece = state.next_pieces[0]
        remaining_pieces = state.next_pieces[1:]
        new_current_piece = Action.move(next_piece, 0, 0)

        state = dataclasses.replace(
            state,
            current_piece=new_current_piece,
            next_pieces=remaining_pieces,
        )

        return dataclasses.replace(state, game_over=state.board.collision(state.current_piece))

    def calculate_score(self) -> "GameState":
        """
        Update the score and level based on lines cleared.
        """
        state = self
        scoring = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        new_score = state.score + scoring.get(state.lines_cleared, 0)
        total_lines = state.lines_cleared + state.lines_cleared
        new_level = state.level

        if total_lines // 10 > state.level - 1:
            new_level += 1

        return dataclasses.replace(state, score=new_score, lines_cleared=total_lines, level=new_level)

    def get_full_board(self) -> Board:
        # Method to get a board with the current piece included, if needed for rendering or other purposes
        return self.board.place_piece(self.current_piece)


@total_ordering
class Action(ABC):
    @abstractmethod
    def apply(self, state: GameState) -> GameState:
        """Apply the action to the game state and return a new game state."""
        pass

    @property
    @abstractmethod
    def priority(self) -> int:
        """Return the priority of the action."""
        pass

    @staticmethod
    def rotate(piece: Piece, direction: Literal["left", "right"], board: Board) -> Piece:
        if direction == "left":
            rotated_shape = np.rot90(piece.shape, -1)
        elif direction == "right":
            rotated_shape = np.rot90(piece.shape, 1)
        else:
            raise ValueError("Invalid rotation direction (btw this should never be called, what dio you do?)")

        wall_kick_offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in wall_kick_offsets:
            new_position = (piece.position[0] + dx, piece.position[1] + dy)
            new_piece = dataclasses.replace(piece, shape=rotated_shape, position=new_position)
            if not board.collision(new_piece):
                new_orientation = (piece.orientation + (1 if direction == "right" else -1)) % 4
                return dataclasses.replace(new_piece, orientation=new_orientation)
        return piece

    @staticmethod
    def move(piece: Piece, dx: int, dy: int) -> Piece:
        new_position = (piece.position[0] + dx, piece.position[1] + dy)
        return dataclasses.replace(piece, position=new_position)

    # Define the less-than comparison based on priority
    def __lt__(self, other: "Action") -> bool:
        return self.priority < other.priority

    # Define equality based on priority
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Action):
            return False
        return self.priority == other.priority


@dataclass(frozen=True)
class MoveLeft(Action):
    @property
    def priority(self) -> int:
        return 1

    def apply(self, state: GameState) -> GameState:
        piece = Action.move(state.current_piece, dx=-1, dy=0)
        if state.board.collision(piece):
            # Undo move by not changing the piece
            return state
        return dataclasses.replace(state, **{"current_piece": piece})


@dataclass(frozen=True)
class MoveRight(Action):
    @property
    def priority(self) -> int:
        return 1

    def apply(self, state: GameState) -> GameState:
        piece = Action.move(state.current_piece, dx=1, dy=0)
        if state.board.collision(piece):
            # Undo move by not changing the piece
            return state
        return dataclasses.replace(state, **{"current_piece": piece})


@dataclass(frozen=True)
class RotateLeft(Action):
    @property
    def priority(self) -> int:
        return 1

    def apply(self, state: GameState) -> GameState:
        piece = Action.rotate(state.current_piece, direction="left", board=state.board)
        if state.board.collision(piece):
            # Undo move by not changing the piece
            return state
        return dataclasses.replace(state, **{"current_piece": piece})


@dataclass(frozen=True)
class RotateRight(Action):
    @property
    def priority(self) -> int:
        return 1

    def apply(self, state: GameState) -> GameState:
        piece = Action.rotate(state.current_piece, direction="right", board=state.board)
        if state.board.collision(piece):
            # Undo move by not changing the piece
            return state
        return dataclasses.replace(state, **{"current_piece": piece})


@dataclass(frozen=True)
class SoftDrop(Action):
    @property
    def priority(self) -> int:
        return 2

    def apply(self, state: GameState) -> GameState:
        piece = Action.move(state.current_piece, dx=0, dy=1)
        if state.board.collision(piece):
            # Undo move by not changing the piece
            return state
        return dataclasses.replace(state, **{"current_piece": piece})


@dataclass(frozen=True)
class HardDrop(Action):
    @property
    def priority(self) -> int:
        return 4  # Highest priority

    def apply(self, state: GameState) -> GameState:
        piece = state.current_piece
        drop_distance = 0
        while not state.board.collision(piece := Action.move(piece, dx=0, dy=1)):
            drop_distance += 1
        # Place the piece
        board_after_lock = Board.place_piece(state.board, piece)
        # Clear lines and update score
        board_after_clear, lines_cleared = Board.clear_lines(board_after_lock)
        # Update score with hard drop bonus
        new_score = state.score + (drop_distance * 2) + {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}.get(lines_cleared, 0)
        total_lines = state.lines_cleared + lines_cleared
        new_level = state.level
        if total_lines // 10 > state.level - 1:
            new_level += 1

        # Get next piece
        if state.next_pieces:
            next_piece = state.next_pieces[0]
            remaining_pieces = state.next_pieces[1:]
            new_current_piece = Action.move(next_piece, 0, 0)  # Ensure position and orientation are set
            # Check for game over condition
            if board_after_clear.collision(new_current_piece):
                return dataclasses.replace(
                    state,
                    board=board_after_clear,
                    score=new_score,
                    lines_cleared=total_lines,
                    level=new_level,
                    game_over=True,
                )
            return dataclasses.replace(
                state,
                board=board_after_clear,
                current_piece=new_current_piece,
                next_pieces=remaining_pieces,
                score=new_score,
                lines_cleared=total_lines,
                level=new_level,
                hold_used=False,
                lock_delay_counter=0,
            )
        else:
            # No more pieces, game over
            return dataclasses.replace(
                state,
                board=board_after_clear,
                score=new_score,
                lines_cleared=total_lines,
                level=new_level,
                game_over=True,
            )


@dataclass(frozen=True)
class Hold(Action):
    @property
    def priority(self) -> int:
        return 5

    def apply(self, state: GameState) -> GameState:
        if state.hold_used:
            return state  # Can't hold again until next piece
        hold_used = True
        if state.held_piece:
            # Swap current piece with held piece
            new_current_piece = Action.move(state.held_piece, 0, 0)  # Ensure position and orientation are set
            new_held_piece = state.current_piece
            return dataclasses.replace(
                state, current_piece=new_current_piece, held_piece=new_held_piece, hold_used=hold_used
            )
        else:
            if state.next_pieces:
                next_piece = state.next_pieces[0]
                remaining_pieces = state.next_pieces[1:]
                new_current_piece = Action.move(next_piece, 0, 0)  # Ensure position and orientation are set
                new_held_piece = state.current_piece
                return dataclasses.replace(
                    state,
                    current_piece=new_current_piece,
                    next_pieces=remaining_pieces,
                    held_piece=new_held_piece,
                    hold_used=hold_used,
                )
            else:
                # No next piece, cannot hold; game over
                return dataclasses.replace(state, game_over=True)


@dataclass(frozen=True)
class Idle(Action):
    @property
    def priority(self) -> int:
        return 0

    def apply(self, state: GameState) -> GameState:
        return state  # Do nothing


class GameAction(Enum):
    MOVE_LEFT = (MoveLeft, auto())
    MOVE_RIGHT = (MoveRight, auto())
    ROTATE_LEFT = (RotateLeft, auto())
    ROTATE_RIGHT = (RotateRight, auto())
    HARD_DROP = (HardDrop, auto())
    SOFT_DROP = (SoftDrop, auto())
    HOLD = (Hold, auto())
    IDLE = (Idle, auto())

    def __init__(self, action_class, index):
        self.action_class = action_class
        self.index = index

    @classmethod
    def from_str(cls, action_name: str) -> "GameAction | None":
        try:
            return cls[action_name.upper()]
        except KeyError:
            return None

    @classmethod
    def is_valid(cls, action_name: str) -> bool:
        return cls.from_str(action_name) is not None

    @classmethod
    def from_index(cls, index: int) -> "GameAction | None":
        return next((action for action in cls if action.index == index), None)

    @cached_property
    @classmethod
    def get_combinations(cls):
        return {action.index: [action.index] for action in cls}


class TetrisEngine:
    def __init__(
        self,
        width,
        height,
        buffer_height,
        lock_delay=0,
        step_reset=False,
        initial_level=1,
        preview_size=4,
        num_lives=10,
        scoring_system: AbstractScoringSystem | None = None,
    ):
        self.width = width
        self.height = height
        self.buffer_height = buffer_height
        self.total_height = self.height + self.buffer_height
        # Initialize the board with buffer zone at the bottom
        self.board = np.zeros(shape=(width, self.total_height, 3), dtype=np.uint8)

        # Use the provided scoring system or create a default one
        self.scoring_system = scoring_system or ScoringSystem({})

        # TODO move this to tetris_shapes so it's dynamic to the actions
        self.value_action_map = {
            0: TetrisEngine.left,  # Move Left
            1: TetrisEngine.right,  # Move Right
            2: TetrisEngine.rotate_left,  # Rotate Left
            3: TetrisEngine.rotate_right,  # Rotate Right
            4: self.hold_swap,  # Hold/Swap
            5: TetrisEngine.hard_drop,  # Hard Drop
            6: TetrisEngine.soft_drop,  # Soft Drop
            7: TetrisEngine.idle,  # Idle
        }

        self.held_piece = None  # No piece is held at the start
        self.held_piece_name = None
        self.hold_used = False
        self.piece_queue = PieceQueue(preview_size)
        self.shape_counts = dict(zip(Piece.SHAPE_NAMES, [0] * len(Piece.SHAPES)))  # TODO do this but for actions
        self.shape = Piece.SHAPES["I"]
        self.shape_name = "I"
        self.anchor = self.get_spawn_position(Piece.SHAPES["I"]["shape"])
        self.actions = [-99]
        self.current_orientation = 0  # Initialize the current orientation
        self.finesse_evaluator = FinesseEvaluator(self.anchor, self.shape_name)
        self.is_current_finesse = True

        self._new_piece()

        self.initial_level = initial_level
        self.level = initial_level
        self.lines_for_next_level = self.level * 10  # Number of lines to clear for next level
        self.gravity_interval = self._calculate_gravity_interval()
        self.gravity_timer = 0

        self.piece_timer = 0

        self.time = 0

        self.score = 0
        self.holes = 0
        self.old_holes = self.holes
        self.piece_height = 0
        self.lines_cleared = 0
        self.lines_cleared_per_step = 0
        self.prev_lines_cleared_per_step = 0

        self.num_lives = num_lives
        self.og_num_lives = num_lives
        self.n_deaths = 0

        self._lock_delay_fn = lambda x: (x + 1) % (max(lock_delay, 0) + 1)
        self._lock_delay = 0
        self._step_reset = step_reset

        self.game_over = False
        self.just_died = False

        self.target_position = None  # Target position for the current piece
        self.planned_actions = []  # Actions planned to reach the target
        self.prev_actions = []

        self.prev_info = {}
        self.prev_info = self.get_info()

    def hold_swap(self, shape, anchor, board, current_orientation, max_orientations=None):
        # TODO make this functional so no side effects can occur to prevent future bugs
        # ! assume that shape, anchor, and board is the same as self
        if self.hold_used:
            return self.shape, self.anchor, current_orientation  # Can't hold/swap again until next piece

        self.hold_used = True

        # Reset the current shape to its default orientation
        while current_orientation != 0:
            shape = TetrisEngine.rotated(shape, cclk=True)
            current_orientation = (current_orientation - 1) % 4

        if self.held_piece is None:  # If no piece is currently held, hold the current piece
            self.held_piece = shape
            self.held_piece_name = self.shape_name
            self._new_piece()  # Generate a new piece to replace the held one
        else:  # Swap the current piece with the held piece
            self.shape, self.held_piece = self.held_piece, shape
            self.shape_name, self.held_piece_name = self.held_piece_name, self.shape_name

        # Reset the anchor to the spawn position
        self.anchor = self.get_spawn_position(self.shape)
        current_orientation = 0  # Reset orientation after hold/swap

        # Update the finesse evaluator with the new piece
        self.finesse_evaluator.reset(self.shape_name, self.anchor)

        self.hold_used = True

        return self.shape, self.anchor, current_orientation

    def _new_piece(self):

        self.shape_name = self.piece_queue.next_piece()
        self.shape_counts[self.shape_name] += 1
        self.shape = Piece.SHAPES[self.shape_name]["shape"]
        self.hold_used = False

        # Use the new spawn position
        self.anchor = self.get_spawn_position(self.shape)
        self.current_orientation = 0  # Reset orientation for new piece

        self.finesse_evaluator.reset(self.shape_name, self.anchor)

    def get_spawn_position(self, shape):
        x_values = [i for i, j in shape]
        min_x, max_x = min(x_values), max(x_values)
        piece_width = max_x - min_x + 1
        spawn_x = (self.width - piece_width) // 2 - min_x
        spawn_y = self.buffer_height - 1
        return (spawn_x, spawn_y)

    def _has_dropped(self):
        return TetrisEngine.is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board)

    def _clear_lines(self):
        # Check each row from top to bottom
        non_zero_cells = np.any(self.board != 0, axis=2)  # Shape: (width, total_height)
        can_clear = np.all(non_zero_cells, axis=0)  # Shape: (total_height,)

        # Rows to keep (not full)
        rows_to_keep = [y for y in range(self.total_height) if not can_clear[y]]

        # Count cleared lines
        cleared_lines = np.sum(can_clear)
        self.prev_lines_cleared_per_step = self.lines_cleared_per_step
        self.lines_cleared_per_step = cleared_lines
        self.lines_cleared += cleared_lines

        # Create new board with cleared lines removed
        new_board = np.zeros_like(self.board)
        j = self.total_height - 1  # Index in new_board

        for y in reversed(rows_to_keep):
            new_board[:, j, :] = self.board[:, y, :]
            j -= 1

        self.board = new_board

        # Update level and gravity if needed
        if self.lines_cleared >= self.lines_for_next_level:
            self.level += 1
            self.lines_for_next_level += 10
            self.gravity_interval = self._calculate_gravity_interval()

        return cleared_lines

    def _count_holes(self, board=None):
        board = board if board is not None else self.board
        # Simplify the board if needed
        if board.ndim == 3:
            board = Board.simplify_board(board)
        elif board.ndim == 2:
            board = board
        else:
            raise ValueError("Invalid board shape. Expected 2D or 3D array.")

        holes = 0

        num_cols, num_rows = board.shape
        for col in range(num_cols):
            col_data = board[col, :]
            filled_indices = np.where(col_data != 0)[0]
            if len(filled_indices) == 0:
                continue  # No filled cells in this column
            highest_filled_row = filled_indices[0]
            holes += np.sum(col_data[highest_filled_row + 1 :] == 0)
        return holes

    def get_info(self):
        simple_board = Board.simplify_board(self.board)
        ghost_piece_anchor = self.get_ghost_piece_position()
        ghost_piece_coords = TetrisEngine._get_coords(self.shape, ghost_piece_anchor, self.width, self.total_height)
        current_piece_coords = TetrisEngine._get_coords(self.shape, self.anchor, self.width, self.total_height)
        float_board_state = TetrisEngine.create_float_board_state(
            simple_board, current_piece_coords, ghost_piece_coords
        )

        settled_board = TetrisEngine.set_piece(
            self.shape_name,
            self.shape,
            self.anchor,
            simple_board,
            self.width,
            self.total_height,
            on=False,
            use_color=False,
        )

        heights = TetrisEngine.get_column_heights(settled_board)
        random_valid_move = self.get_random_valid_move()  # Include the random move in the info dict

        info = {
            "time": self.time,
            "current_piece": self.shape_name,
            "current_shape": self.shape,
            "anchor": (self.anchor[0], 40 - self.anchor[1]),
            "current_piece_coords": current_piece_coords,
            "ghost_piece_anchor": (ghost_piece_anchor[0], 40 - ghost_piece_anchor[1]),
            "ghost_piece_coords": ghost_piece_coords,
            "score": self.score,
            "total_lines_cleared": self.lines_cleared,
            "lines_cleared_per_step": self.lines_cleared_per_step,
            "prev_lines_cleared_per_step": self.prev_lines_cleared_per_step,
            "holes": self.holes,
            "old_holes": self.old_holes,
            "deaths": self.n_deaths,
            "lives_left": self.num_lives - self.n_deaths,
            "lost_a_life": self.just_died,
            "statistics": self.shape_counts,
            "level": self.level,
            "gravity_interval": self.gravity_interval,
            "gravity_timer": self.gravity_timer,
            "piece_timer": self.piece_timer,
            "next_piece": self.piece_queue.get_preview(),
            "held_piece": self.held_piece,
            "held_piece_name": self.held_piece_name,
            "prev_info": self.prev_info,
            "actions": self.actions,
            "hold_used": self.hold_used,
            "settled_board": settled_board,
            "float_board_state": float_board_state,
            "simple_board": simple_board,
            "heights": heights,
            "bumpiness": np.sum(np.abs(np.diff(heights))),
            "agg_height": np.sum(heights),
            "game_over": self.game_over,
            "is_current_finesse": self.is_current_finesse,
            "is_finesse_complete": self.finesse_evaluator.finesse_complete,
            "random_valid_move": (random_valid_move),  # Include the random move in the info dict
            "random_valid_move_str": GameAction.from_index(random_valid_move),
            # "current_finesse_score": self.current_finesse_score,
        }

        self.just_died = False

        self.prev_info = info

        return info

    def _calculate_gravity_interval(self):
        if self.level == 0:
            return float("inf")
        base_interval = 60  # Starting interval for level 1
        difficulty_factor = 0.2  # Adjust this to control difficulty progression
        return max(1, base_interval * math.exp(-difficulty_factor * (self.level - 1)))

    def step(self, actions):
        # Action priorities: lower numbers have higher priority
        action_priority = {
            0: 1,  # Move Left
            1: 1,  # Move Right
            2: 1,  # Rotate Left
            3: 1,  # Rotate Right
            4: 2,  # Hold/Swap
            5: 3,  # Hard Drop (highest priority)
            6: 2,  # Soft Drop
            7: 0,  # Idle (lowest priority)
        }

        actions = sorted(actions, key=lambda action: action_priority[action])
        self.actions = actions

        current_orientation = self.current_orientation  # Start with the current orientation

        # Process each action in the sorted order
        for action in actions:

            self.anchor = (int(self.anchor[0]), int(self.anchor[1]))
            action_method = self.value_action_map[action]
            max_orientations = self.finesse_evaluator.get_max_orientations(self.shape_name)

            self.shape, self.anchor, current_orientation = action_method(
                self.shape,
                self.anchor,
                self.board,
                current_orientation=current_orientation,
                max_orientations=max_orientations,
            )

            # Update the evaluator
            self.finesse_evaluator.evaluate_action(action)

        self.current_orientation = current_orientation  # Update the engine's orientation

        self.time += 1
        self.gravity_timer += 1
        self.piece_timer += 1
        reward = self.scoring_system.calculate_step_reward()

        done = False
        cleared_lines = 0

        if (
            GameAction.from_str("hard_drop") in actions
            or self.gravity_timer >= self.gravity_interval
            and self.gravity_interval != float("inf")
        ):
            self.gravity_timer = 0
            self.shape, new_anchor, current_orientation = TetrisEngine.soft_drop(self.shape, self.anchor, self.board)
            if self._step_reset and (self.anchor != new_anchor):
                self._lock_delay = 0

            self.anchor = new_anchor

            if self._has_dropped() or 5 in actions:
                self.piece_timer = 0
                self._lock_delay = self._lock_delay_fn(self._lock_delay)

                if self._lock_delay == 0:
                    self._set_piece(True)
                    cleared_lines = self._clear_lines()

                    reward += self.scoring_system.calculate_clear_reward(cleared_lines)
                    self.score += self.scoring_system.calculate_clear_reward(cleared_lines)

                    self.game_over = False
                    self.old_holes = self.holes
                    # self.old_height = self.piece_height
                    self.holes = self._count_holes()
                    if np.any(self.board[:, : self.buffer_height]):
                        self.game_over = True
                    else:
                        new_height = sum(np.any(self.board, axis=0))

                        # reward += self.scoring_system.calculate_height_penalty(self.board)
                        # reward += self.scoring_system.calculate_height_increase_penalty(new_height, self.old_height)
                        # reward += self.scoring_system.calculate_holes_penalty(self.holes)
                        # reward += self.scoring_system.calculate_holes_increase_penalty(self.holes, self.old_holes)

                        self.piece_height = new_height
                        self.finesse_evaluator.piece_placed(self.anchor, self.current_orientation)
                        self._new_piece()

                        if TetrisEngine.is_occupied(self.shape, self.anchor, self.board):

                            self.game_over = True
                    if self.game_over:
                        self.n_deaths += 1
                        reward = -100

                        if self.n_deaths > self.num_lives:
                            done = True
                        else:
                            self.just_died = True
                            self.clear()  # TODO having clear() called here is messing with the reward calculation with losing a life
        else:
            self.is_current_finesse = self.finesse_evaluator.evaluate_finesse(self.anchor, self.current_orientation)

        self._set_piece(True)
        state = np.copy(self.board)  # Ensure state being returned contains the current piece
        self._set_piece(False)

        self.score = self.evaluate_board(self.board)
        return state, self.score, done

    def clear(self):
        self.score = 0
        self.holes = 0
        self.piece_height = 0
        self.old_holes = self.holes
        self.lines_cleared = 0
        self._new_piece()

        self.board = np.zeros_like(self.board)
        self.num_lives = self.og_num_lives

        self.level = self.initial_level
        self.lines_for_next_level = 10
        self.prev_lines_cleared_per_step = 0
        self.lines_cleared_per_step = 0
        self.gravity_interval = self._calculate_gravity_interval()
        self.gravity_timer = 0
        self.piece_timer = 0
        self.target_position = None  # Target position for the current piece
        self.planned_actions = []  # Actions planned to reach the target
        self.prev_actions = []

        self.prev_info = {}
        self.prev_info = self.get_info()

        return self.board

    def reset(self):

        self.time = 0
        self.score = 0
        self.holes = 0
        self.lines_cleared = 0
        self.prev_lines_cleared_per_step = 0
        self.lines_cleared_per_step = 0
        self.piece_height = 0
        self.old_holes = self.holes
        self.n_deaths = 0
        self._new_piece()

        self.board = np.zeros_like(self.board)
        self.game_over = False

        self.level = self.initial_level
        self.lines_for_next_level = 10
        self.gravity_interval = self._calculate_gravity_interval()
        self.gravity_timer = 0
        self.piece_timer = 0
        self.target_position = None  # Target position for the current piece
        self.planned_actions = []  # Actions planned to reach the target
        self.prev_actions = []

        self.prev_info = {}
        self.prev_info = self.get_info()

        return self.board

    def get_ghost_piece_position(self):
        ghost_anchor = self.anchor
        while not TetrisEngine.is_occupied(self.shape, ghost_anchor, self.board):
            ghost_anchor = (ghost_anchor[0], ghost_anchor[1] + 1)
        # Move back up one row to the last valid position
        ghost_anchor = (ghost_anchor[0], ghost_anchor[1] - 1)
        return ghost_anchor

    def render(self):
        self._set_piece(True)
        state = np.copy(self.board)
        self._set_piece(False)

        # Add ghost piece
        ghost_anchor = self.get_ghost_piece_position()

        ghost_color = tuple(min(255, c + 70) for c in Piece.SHAPES[self.shape_name]["color"])  # Lighter color

        return state, self.shape, ghost_anchor, ghost_color

    def _set_piece(self, on=False):
        """Set the current piece on or off in the game state.

        Parameters
        ----------
        on : bool, optional
            Whether to set the piece on or off. Defaults to False.
        """
        self.board = TetrisEngine.set_piece(
            self.shape_name, self.shape, self.anchor, self.board, self.width, self.total_height, on=on
        )

    @staticmethod
    def set_piece(shape_name, shape, anchor, in_board, width, total_height, on=False, use_color=True):
        """
        Set the given shape on or off in the given game state.

        Parameters
        ----------
        shape_name : str
            The name of the shape to set.
        shape : List[Tuple[int, int]]
            The shape to set.
        anchor : Tuple[int, int]
            The anchor position of the shape.
        in_board : np.ndarray
            The game state to modify.
        width : int
            The width of the game state.
        total_height : int
            The total height of the game state.
        on : bool, optional
            Whether to set the shape on or off. Defaults to False.
        use_color : bool, optional
            Whether to use the color of the shape. Defaults to True.

        Returns
        -------
        np.ndarray
            The modified game state.
        """
        board = np.copy(in_board)

        if shape_name:
            if use_color:
                color = Piece.SHAPES[shape_name]["color"] if on else (0, 0, 0)
            else:
                color = 1 if on else 0
            for i, j in shape:
                x, y = int(anchor[0] + i), int(anchor[1] + j)
                if 0 <= x < width and 0 <= y < total_height:
                    board[x, y] = color

        return board

    @staticmethod
    def _get_coords(shape, anchor, width, total_height):
        """Get the coordinates of the shape on the board.

        Parameters
        ----------
        shape : List[Tuple[int, int]]
            The shape to get coordinates for.
        anchor : Tuple[int, int]
            The anchor position of the shape.
        width : int
            The width of the game state.
        total_height : int
            The total height of the game state.

        Returns
        -------
        List[Tuple[int, int]]
            The coordinates of the shape on the board.
        """
        coords_list = []
        for i, j in shape:
            x, y = int(anchor[0] + i), int(anchor[1] + j)
            if 0 <= x < width and 0 <= y < total_height:
                coords_list.append((x, 40 - y))
        return coords_list

    def __repr__(self):
        self._set_piece(True)
        s = "o" + "-" * self.width + "o\n"
        s += "\n".join(["|" + "".join(["X" if j.any() else " " for j in i]) + "|" for i in self.board.T])
        s += "\no" + "-" * self.width + "o"
        self._set_piece(False)
        return s

    @staticmethod
    def create_float_board_state(board, current_piece_coords, ghost_piece_coords):
        """
        Create a float board state with:
        0 for no blocks
        0.1 for ghost piece blocks
        0.5 for placed blocks
        1 for current piece blocks

        Args:
        board (np.array): The current board state (usually binary)
        current_piece_coords (list of tuples): Coordinates of the current piece
        ghost_piece_coords (list of tuples): Coordinates of the ghost piece

        Returns:
        np.array: Float board state
        """
        # Initialize the float board with zeros
        float_board = np.zeros_like(board, dtype=np.float32)

        # Set placed blocks to 0.5
        float_board[board > 0] = 0.5

        # Set ghost piece blocks to 0.1
        for x, y in ghost_piece_coords:
            if 0 <= x < float_board.shape[0] and 0 <= y < float_board.shape[1]:
                if float_board[x, y] == 0:  # Only set if the cell is empty
                    float_board[x, y] = 0.1

        # Set current piece blocks to 1
        for x, y in current_piece_coords:
            if 0 <= x < float_board.shape[0] and 0 <= y < float_board.shape[1]:
                float_board[x, y] = 1.0

        return float_board

    def _count_lines_to_clear(self, board):
        """
        Count how many lines can be cleared with the current board.
        """
        non_zero_cells = np.any(board != 0, axis=1)
        can_clear = np.all(non_zero_cells, axis=0)  # Shape: (total_height,)
        return np.sum(can_clear)

    def evaluate_board(self, board):
        # Simplify the board if needed
        if board.ndim == 3:
            board = Board.simplify_board(board)

        holes = self._count_holes(board)
        heights = TetrisEngine.get_column_heights(board)
        aggregate_height = np.sum(heights)
        bumpiness = np.sum(np.abs(np.diff(heights)))
        lines_cleared = self._count_lines_to_clear(board)
        well_sums = np.sum(self.calculate_well_sums(heights))

        # Heuristic scoring
        # Using coefficients from well-known Tetris AI heuristics
        score = 0
        score += lines_cleared * 0.760666
        score -= aggregate_height * 0.510066
        score -= holes * 0.35663
        score -= bumpiness * 0.184483
        score -= well_sums * 0.1  # Penalty for wells

        return score

    def _choose_target_position(self):
        """
        Choose a target position on the board where the current piece should be placed.
        """
        # Generate all possible final positions for the current piece
        placements = self._generate_possible_placements()
        scored_placements = []
        # Evaluate each possible placement using a heuristic
        for placement in placements:
            simulated_board = self.simulate_placement(placement, self.board)
            score = self.evaluate_board(simulated_board)
            scored_placements.append((score, placement))

        # Sort placements by score in descending order
        scored_placements.sort(reverse=True, key=lambda x: x[0])

        # Select the top N placements
        top_N = 1
        top_placements = [placement for score, placement in scored_placements[:top_N]]

        # print(f"{self.time=}")
        # print(f"{scored_placements=}")
        # print(f"{top_placements=}")

        if not top_placements:
            # No valid placements found
            self.target_position = None
            return

        # Randomly choose one from the top placements
        self.target_position = random.choice(top_placements)

    def _generate_possible_placements(self):
        """
        Generate all possible final positions (placements) for the current piece.
        """
        placements = []
        board = Board.simplify_board(self.board)

        for orientation in range(max_orientations := self.finesse_evaluator.get_max_orientations(self.shape_name)):
            # Rotate the shape to the desired orientation
            rotated_shape = self.shape
            for _ in range((orientation - self.current_orientation) % max_orientations):
                rotated_shape = TetrisEngine.rotated(rotated_shape, cclk=True)

            # Get the min and max x positions for the rotated shape
            min_i = min([i for i, j in rotated_shape])
            max_i = max([i for i, j in rotated_shape])
            min_x = -min_i
            max_x = self.width - max_i - 1

            for x in range(min_x, max_x + 1):
                y = 0
                # Drop the piece down until it cannot go further
                while not TetrisEngine.is_occupied(rotated_shape, (x, y + 1), board):
                    y += 1
                # Record the placement
                placement = {"x": x, "y": y, "orientation": orientation, "shape": rotated_shape}
                # if self.is_placement_acceptable(placement, board):
                placements.append(placement)

        return placements

    def is_placement_acceptable(self, placement, board):
        """
        Check if a placement is acceptable, taking into account the overall board state.

        Returns True if acceptable, False otherwise.
        """
        # Simulate the placement
        # prev_max_heights = TetrisEngine.get_column_heights(self.board)
        # simulated_board = self.simulate_placement(placement, board)
        # simplified_board = simplify_board(simulated_board)
        # heights = TetrisEngine.get_column_heights(simplified_board)

        # Compute a measure of board condition
        # aggregate_height = np.sum(heights)
        # num_holes = self._count_holes(simplified_board)

        # # Adjust thresholds based on board condition
        # base_max_height_diff = 4
        # base_max_well_depth = 4

        # # Calculate 'badness' score
        # badness = aggregate_height + num_holes * 10  # Adjust weights as appropriate

        # # Define threshold for poor board condition
        # badness_threshold = 100  # Adjust as appropriate

        # if badness > badness_threshold:
        #     # Relax constraints if board is in poor condition
        #     max_height_diff = base_max_height_diff + 3
        #     max_well_depth = base_max_well_depth + 3
        # else:
        #     max_height_diff = base_max_height_diff
        #     max_well_depth = base_max_well_depth

        # # Compute differences between adjacent columns
        # height_diffs = np.abs(np.diff(heights))

        # # Avoid placements with high column height differences
        # if np.any(height_diffs > max_height_diff):
        #     return False

        # # Avoid placements that create deep wells
        # well_sums = self.calculate_well_sums(heights)
        # if np.any(well_sums > max_well_depth):
        #     return False
        # if np.max(heights) > 10 and not np.max(prev_max_heights) > 10:
        #     return False
        return True

    def is_placement_acceptable_bruh(self, placement, board):
        """
        Check if a placement is acceptable, i.e., it doesn't place the piece on top of narrow towers
        or create overhangs.

        Returns True if acceptable, False otherwise.
        """
        # # Simulate the placement
        # simulated_board = self.simulate_placement(placement, board)
        # heights = TetrisEngine.get_column_heights(simplify_board(simulated_board))
        # # Compute the differences between adjacent columns
        # height_diffs = np.abs(np.diff(heights))

        # # Avoid placements that result in high differences between adjacent columns
        # max_height_diff = 3  # Threshold can be adjusted
        # if np.any(height_diffs > max_height_diff):
        #     return False

        # # Avoid placements that create deep wells
        # well_sums = self.calculate_well_sums(heights)
        # max_well_depth = 3
        # if np.any(well_sums > max_well_depth):
        #     return False

        return True

    def calculate_well_sums(self, heights):
        """
        Calculate the well sums of the board.
        """
        wells = np.zeros_like(heights)
        for i in range(len(heights)):
            left = heights[i - 1] if i > 0 else heights[i]
            right = heights[i + 1] if i < len(heights) - 1 else heights[i]
            if heights[i].any() < left and heights[i].any() < right:
                wells[i] = min(left, right) - heights[i]
        return wells

    def _find_path_to_target(self):
        if self.target_position is None:
            return []

        moves = []

        # Get the current position and orientation of the piece
        current_x, current_y, current_orientation = self.anchor[0], self.anchor[1], self.current_orientation
        target_x, target_y, target_orientation = (
            self.target_position["x"],
            self.target_position["y"],
            self.target_position["orientation"],
        )

        # Calculate how much to move left or right
        x_diff = target_x - current_x
        if x_diff > 0:
            moves.extend([GameAction.from_str("ROTATE_RIGHT")] * x_diff)  # Move right
        elif x_diff < 0:
            moves.extend([GameAction.from_str("ROTATE_LEFT")] * abs(x_diff))  # Move left

        # Calculate the minimal number of rotations needed to match the target orientation
        orientation_diff = (target_orientation - current_orientation) % 4
        if orientation_diff == 1:
            moves.append(GameAction.from_str("ROTATE_RIGHT"))  # Rotate right once
        elif orientation_diff == 3:
            moves.append(GameAction.from_str("ROTATE_LEFT"))  # Rotate left once
        elif orientation_diff == 2:
            # Rotate twice in either direction
            moves.extend([GameAction.from_str("ROTATE_RIGHT")] * 2)  # Rotate right twice

        # After the moves and rotations, perform a hard drop
        moves.append(GameAction.from_str("HARD_DROP"))

        return moves

    def get_shape_at_orientation(self, shape_name, orientation):
        shape = Piece.SHAPES[shape_name]["shape"]
        for _ in range(orientation % 4):
            shape = TetrisEngine.rotated(shape, cclk=True)  # Rotate counter-clockwise
        return shape

    def get_lowest_y(self, x, orientation, shape):
        y = 0
        while not TetrisEngine.is_occupied(shape, (x, y + 1), Board.simplify_board(self.board)):
            y += 1
        return y

    def _heuristic_cost(self, state, target_state):
        x, y, orientation = state
        target_x, target_y, target_orientation = target_state
        dx = abs(x - target_x)
        dy = abs(y - target_y)
        d_orientation = min(abs(orientation - target_orientation), 4 - abs(orientation - target_orientation))
        return dx + dy + d_orientation

    def get_random_valid_move(self):
        """
        Choose a random optimal spot for the current piece and work out the least amount of inputs to reach it.
        """

        if self.game_over:
            # print("uh oh, game over")
            return random.choice([GameAction.from_str("MOVE_LEFT"), GameAction.from_str("MOVE_RIGHT")])
        # If we have no planned actions, we need to choose a new target and plan
        if not self.planned_actions:
            self._choose_target_position()
            # print(f"{self.target_position=}")
            self.planned_actions = self._find_path_to_target()
            # print(f"{self.planned_actions=}")
            self.prev_actions = []

        if self.planned_actions:
            # Get the next action in the planned sequence
            action = self.planned_actions.pop(0)
            self.prev_actions.append(action)
            # print(f"yaaaassssss: {action=}")
            return action
        else:
            # If no actions are planned, do this I guess
            return random.choice(
                [
                    random.choice([GameAction.from_str("MOVE_LEFT"), GameAction.from_str("MOVE_RIGHT")]),
                    GameAction.from_str("HOLD"),
                ]
            )

    def simulate_placement(self, placement, board):
        """
        Simulate placing the piece at the given placement and return the resulting board.
        """
        # Copy the current board
        board_copy = np.copy(board)
        # Place the piece
        anchor = (placement["x"], placement["y"])
        board_copy = TetrisEngine.set_piece(
            self.shape_name,
            placement["shape"],
            anchor,
            board_copy,
            self.width,
            self.total_height,
            on=True,
            use_color=False,
        )
        # Simulate clearing lines
        board_copy = self.simulate_clear_lines(board_copy)
        return board_copy

    def simulate_clear_lines(self, board):
        """
        Simulate clearing lines in the board.
        """
        non_zero_cells = board != 0
        can_clear = np.all(non_zero_cells, axis=0)
        rows_to_keep = [y for y in range(board.shape[1]) if not can_clear[y].all()]
        new_board = np.zeros_like(board)
        j = board.shape[1] - 1  # Index in new_board

        for y in reversed(rows_to_keep):
            new_board[:, j] = board[:, y]
            j -= 1

        return new_board
