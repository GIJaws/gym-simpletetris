import random
import numpy as np
from gym_simpletetris.tetris.finesse_evaluator import FinesseEvaluator
from gym_simpletetris.tetris.scoring_system import AbstractScoringSystem, ScoringSystem
from gym_simpletetris.tetris.tetris_shapes import (
    ACTION_NAME_TO_INDEX,
    SHAPE_NAMES,
    SHAPES,
    simplify_board,
    BASIC_ACTIONS,
    ACTION_COMBINATIONS,
)
import math
import heapq


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
        self.shape_counts = dict(zip(SHAPE_NAMES, [0] * len(SHAPES)))  # TODO do this but for actions
        self.shape = SHAPES["I"]
        self.shape_name = "I"
        self.anchor = self.get_spawn_position(SHAPES["I"]["shape"])
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
        self.shape = SHAPES[self.shape_name]["shape"]
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
            board = simplify_board(board)
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
        simple_board = simplify_board(self.board)
        ghost_piece_anchor = self.get_ghost_piece_position()
        lines_cleared_per_step = self.lines_cleared - self.prev_info.get("lines_cleared", 0)
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
            "lines_cleared_per_step": lines_cleared_per_step,
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
            "random_valid_move_str": BASIC_ACTIONS[random_valid_move],
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
            ACTION_NAME_TO_INDEX["hard_drop"] in actions
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
        return state, reward, done

    def clear(self):
        self.score = 0
        self.holes = 0
        self.piece_height = 0
        self.old_holes = self.holes

        self._new_piece()

        self.board = np.zeros_like(self.board)
        self.num_lives = self.og_num_lives

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

    def reset(self):

        self.time = 0
        self.score = 0
        self.holes = 0
        self.lines_cleared = 0
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

        ghost_color = tuple(min(255, c + 70) for c in SHAPES[self.shape_name]["color"])  # Lighter color

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
                color = SHAPES[shape_name]["color"] if on else (0, 0, 0)
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
    def rotated(shape, cclk=False):
        """Rotate a shape clockwise (cclk=False) or counterclockwise (cclk=True).

        Parameters
        ----------
        shape : List[Tuple[int, int]]
            The shape to rotate.
        cclk : bool, optional
            Whether to rotate clockwise or counterclockwise. Defaults to False.

        Returns
        -------
        List[Tuple[int, int]]
            The rotated shape.
        """
        if cclk:
            return [(-j, i) for i, j in shape]
        else:
            return [(j, -i) for i, j in shape]

    @staticmethod
    def is_occupied(shape, anchor, board):
        for i, j in shape:
            x = anchor[0] + i
            y = anchor[1] + j
            if x < 0 or x >= board.shape[0] or y < 0 or y >= board.shape[1]:
                return True  # Out of bounds
            if np.any(board[x, y]):
                return True  # Position already occupied
        return False

    @staticmethod
    def left(shape, anchor, board, current_orientation=None, max_orientations=None):
        new_anchor = (anchor[0] - 1, anchor[1])
        if TetrisEngine.is_occupied(shape, new_anchor, board):
            return shape, anchor, current_orientation
        else:
            return shape, new_anchor, current_orientation

    @staticmethod
    def right(shape, anchor, board, current_orientation=None, max_orientations=None):
        new_anchor = (anchor[0] + 1, anchor[1])
        if TetrisEngine.is_occupied(shape, new_anchor, board):
            return shape, anchor, current_orientation
        else:
            return shape, new_anchor, current_orientation

    @staticmethod
    def soft_drop(shape, anchor, board, current_orientation=None, max_orientations=None):
        new_anchor = (anchor[0], anchor[1] + 1)
        if TetrisEngine.is_occupied(shape, new_anchor, board):
            return shape, anchor, current_orientation
        else:
            return shape, new_anchor, current_orientation

    @staticmethod
    def hard_drop(shape, anchor, board, current_orientation=None, max_orientations=None):
        while not TetrisEngine.is_occupied(shape, (anchor[0], anchor[1] + 1), board):
            anchor = (anchor[0], anchor[1] + 1)
        return shape, anchor, current_orientation

    @staticmethod
    def rotate_left(shape, anchor, board, current_orientation=None, max_orientations=None):
        new_shape = TetrisEngine.rotated(shape, cclk=False)
        if TetrisEngine.is_occupied(new_shape, anchor, board):
            return shape, anchor, current_orientation
        else:
            if current_orientation is not None and max_orientations is not None:
                current_orientation = (current_orientation - 1) % max_orientations
            return new_shape, anchor, current_orientation

    @staticmethod
    def rotate_right(shape, anchor, board, current_orientation=None, max_orientations=None):
        new_shape = TetrisEngine.rotated(shape, cclk=True)
        if TetrisEngine.is_occupied(new_shape, anchor, board):
            return shape, anchor, current_orientation
        else:
            if current_orientation is not None and max_orientations is not None:
                current_orientation = (current_orientation + 1) % max_orientations
            return new_shape, anchor, current_orientation

    @staticmethod
    def idle(shape, anchor, board, current_orientation=None, max_orientations=None):
        return shape, anchor, current_orientation

    @staticmethod
    def get_column_heights(board):
        non_zero_mask = board != 0
        heights = board.shape[1] - np.argmax(non_zero_mask, axis=1)
        return np.where(non_zero_mask.any(axis=1), heights, 0)

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
            board = simplify_board(board)

        holes = self._count_holes(board)
        heights = TetrisEngine.get_column_heights(board)
        aggregate_height = np.sum(heights)
        bumpiness = np.sum(np.abs(np.diff(heights)))
        lines_cleared = self._count_lines_to_clear(board)
        # well_sums = np.sum(self.calculate_well_sums(heights))

        # Heuristic scoring
        # Using coefficients from well-known Tetris AI heuristics
        score = 0
        score += lines_cleared * 0.760666
        score -= aggregate_height * 0.510066
        score -= holes * 0.35663
        score -= bumpiness * 0.184483
        # score -= well_sums * 0.1  # Penalty for wells

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
        top_N = 5
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
        board = simplify_board(self.board)

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
                # placement = {"x": x, "y": (40 - y) * (y != 0), "orientation": orientation, "shape": rotated_shape}
                placement = {"x": x, "y": y, "orientation": orientation, "shape": rotated_shape}
                if self.is_placement_acceptable(placement, board):
                    placements.append(placement)

        return placements

    def is_placement_acceptable(self, placement, board):
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

        moves = [
            (ACTION_NAME_TO_INDEX["left"], "left"),
            (ACTION_NAME_TO_INDEX["right"], "right"),
            (ACTION_NAME_TO_INDEX["rotate_left"], "rotate_left"),
            (ACTION_NAME_TO_INDEX["rotate_right"], "rotate_right"),
        ]

        start_state = (self.anchor[0], self.anchor[1], self.current_orientation)
        target_state = (self.target_position["x"], self.target_position["y"], self.target_position["orientation"])

        open_set = []
        heapq.heappush(open_set, (0, start_state, []))
        visited = set()

        while open_set:
            _, (x, y, orientation), actions = heapq.heappop(open_set)

            if (x, y, orientation) in visited:
                continue
            visited.add((x, y, orientation))

            if (x, y, orientation) == target_state:
                return actions + [ACTION_NAME_TO_INDEX["hard_drop"]]

            # Simulate gravity
            if not TetrisEngine.is_occupied(
                self.get_shape_at_orientation(self.shape_name, orientation), (x, y + 1), simplify_board(self.board)
            ):
                y += 1

            for action_value, action_name in moves:
                new_orientation = orientation
                new_x = x
                new_y = y

                if action_name == "left":
                    new_x = x - 1
                elif action_name == "right":
                    new_x = x + 1
                elif action_name == "rotate_left":
                    new_orientation = (orientation - 1) % 4
                elif action_name == "rotate_right":
                    new_orientation = (orientation + 1) % 4
                elif action_name == "soft_drop":
                    new_y = y + 1

                # Check boundaries and collisions
                shape_at_new_orientation = self.get_shape_at_orientation(self.shape_name, new_orientation)
                if TetrisEngine.is_occupied(shape_at_new_orientation, (new_x, new_y), simplify_board(self.board)):
                    continue

                new_state = (new_x, new_y, new_orientation)
                cost_to_new_state = len(actions) + 1
                estimated_cost = cost_to_new_state + self._heuristic_cost(new_state, target_state)
                heapq.heappush(open_set, (estimated_cost, new_state, actions + [action_value]))

        return []

    def get_shape_at_orientation(self, shape_name, orientation):
        shape = SHAPES[shape_name]["shape"]
        for _ in range(orientation % 4):
            shape = TetrisEngine.rotated(shape, cclk=True)  # Rotate counter-clockwise
        return shape

    def get_lowest_y(self, x, orientation, shape):
        y = 0
        while not TetrisEngine.is_occupied(shape, (x, y + 1), simplify_board(self.board)):
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
            print("uh oh, game over")
            return random.choice([ACTION_NAME_TO_INDEX["left"], ACTION_NAME_TO_INDEX["right"]])
        # If we have no planned actions, we need to choose a new target and plan
        if not self.planned_actions:
            self._choose_target_position()
            print(f"{self.target_position=}")
            self.planned_actions = self._find_path_to_target()
            print(f"{self.planned_actions=}")
            self.prev_actions = []

        if self.planned_actions:
            # Get the next action in the planned sequence
            action = self.planned_actions.pop(0)
            self.prev_actions.append(action)
            print(f"yaaaassssss: {action=}")
            return action
        else:
            # If no actions are planned, default to idle
            return random.choice([ACTION_NAME_TO_INDEX["left"], ACTION_NAME_TO_INDEX["right"]])

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


class PieceQueue:
    def __init__(self, preview_size=4):
        self.preview_size = max(1, preview_size)  # Ensure at least 1 piece in preview
        self.pieces = []
        self.bag = []
        self.refill_bag()
        self.fill_queue()

    def refill_bag(self):
        self.bag = list(SHAPE_NAMES)
        random.shuffle(self.bag)

    def next_piece(self):
        if len(self.pieces) <= self.preview_size:
            self.fill_queue()
        return self.pieces.pop(0)

    def fill_queue(self):
        while len(self.pieces) < self.preview_size * 2:  # Keep 2x preview size in queue
            if not self.bag:
                self.refill_bag()
            self.pieces.append(self.bag.pop())

    def get_preview(self):
        while len(self.pieces) < self.preview_size:
            self.fill_queue()
        return self.pieces[: self.preview_size]
