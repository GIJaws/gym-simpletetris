import random
import numpy as np
from gym_simpletetris.tetris.scoring_system import AbstractScoringSystem, ScoringSystem
from gym_simpletetris.tetris.tetris_shapes import SHAPE_NAMES, SHAPES, simplify_board
import math


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
        self.shape = None
        self.shape_name = None
        self.anchor = (np.nan, np.nan)
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

        self.actions = [-99]
        self.game_over = False
        self.just_died = False

        self.prev_info = {}
        self.prev_info = self.get_info()

    def hold_swap(self, shape, anchor, board):
        # TODO make this functional so no side effects can occur to prevent future bugs
        # ! assume that shape, anchor, and board is the same as self

        if self.hold_used:
            return self.shape, self.anchor  # Can't hold/swap again until next piece

        self.hold_used = True

        if self.held_piece is None:  # If no piece is currently held, hold the current piece
            self.held_piece = self.shape
            self.held_piece_name = self.shape_name
            self._new_piece()  # Generate a new piece to replace the held one
            self.hold_used = True  # So the player can't switch back again
        else:  # Swap the current piece with the held piece
            self.shape, self.held_piece = self.held_piece, self.shape
            self.shape_name, self.held_piece_name = self.held_piece_name, self.shape_name

        # Reset the anchor to the spawn position
        self.anchor = self.get_spawn_position(self.shape)
        return self.shape, self.anchor

    def _new_piece(self):
        # Spawn in the middle of the width, at the top of the height area
        # self.anchor = (self.width // 2, self.buffer_height - 1)
        self.shape_name = self.piece_queue.next_piece()
        self.shape_counts[self.shape_name] += 1
        self.shape = SHAPES[self.shape_name]["shape"]
        self.hold_used = False

        # Use the new spawn position
        self.anchor = self.get_spawn_position(self.shape)

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

    def _count_holes(self):
        # TODO check if board needs to be simplified instead of always simplifying
        if self.board.ndim == 3:
            board = simplify_board(self.board)
        elif self.board.ndim == 2:
            board = self.board
        else:
            raise ValueError("Invalid board shape. Expected 2D or 3D array.")

        holes = 0

        num_cols, num_rows = board.shape
        for col in range(num_cols):
            block_found = False
            for row in range(num_rows):
                cell = board[col, row]
                if cell != 0:
                    block_found = True
                elif block_found and cell == 0:
                    holes += 1
        return holes

    def get_info(self):
        simple_board = simplify_board(self.board)
        ghost_piece_anchor = self.get_ghost_piece_position()
        lines_cleared_per_step = self.lines_cleared - self.prev_info.get("lines_cleared", 0)
        ghost_piece_coords = TetrisEngine._get_coords(self.shape, ghost_piece_anchor, self.width, self.total_height)
        current_piece_coords = TetrisEngine._get_coords(self.shape, ghost_piece_anchor, self.width, self.total_height)
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

        heights = (TetrisEngine.get_column_heights(settled_board),)

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
            "heights": TetrisEngine.get_column_heights(settled_board),
            "agg_height": np.sum(heights) / 200,
            "game_over": self.game_over,
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

        self.actions = actions

        # Process each action in the sorted order
        for action in sorted(actions, key=lambda action: action_priority[action]):
            self.anchor = (int(self.anchor[0]), int(self.anchor[1]))
            self.shape, self.anchor = self.value_action_map[action](self.shape, self.anchor, self.board)

        self.time += 1
        self.gravity_timer += 1
        self.piece_timer += 1
        reward = self.scoring_system.calculate_step_reward()

        done = False
        cleared_lines = 0

        if 5 in actions or self.gravity_timer >= self.gravity_interval and self.gravity_interval != float("inf"):
            self.gravity_timer = 0
            self.shape, new_anchor = TetrisEngine.soft_drop(self.shape, self.anchor, self.board)
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
                    self.old_height = self.piece_height
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

        # print(f"{ghost_anchor=}, {self.anchor=}")

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
        s += "\n".join(["|" + "".join(["X" if j else " " for j in i]) + "|" for i in self.board.T])
        s += "\no" + "-" * self.width + "o"
        self._set_piece(False)
        return s

    @staticmethod
    def rotated(shape, cclk=False):
        if cclk:
            return [(-j, i) for i, j in shape]
        else:
            return [(j, -i) for i, j in shape]

    @staticmethod
    def is_occupied(shape, anchor, board):
        for i, j in shape:
            x = anchor[0] + i
            y = anchor[1] + j
            if x < 0 or x >= board.shape[0] or y >= board.shape[1]:
                return True  # Out of bounds
            if np.any(board[x, y]):
                return True  # Position already occupied
        return False

    @staticmethod
    def left(shape, anchor, board):
        new_anchor = (anchor[0] - 1, anchor[1])
        return (shape, anchor) if TetrisEngine.is_occupied(shape, new_anchor, board) else (shape, new_anchor)

    @staticmethod
    def right(shape, anchor, board):
        new_anchor = (anchor[0] + 1, anchor[1])

        return (shape, anchor) if TetrisEngine.is_occupied(shape, new_anchor, board) else (shape, new_anchor)

    @staticmethod
    def soft_drop(shape, anchor, board):
        new_anchor = (anchor[0], anchor[1] + 1)
        return (shape, anchor) if TetrisEngine.is_occupied(shape, new_anchor, board) else (shape, new_anchor)

    @staticmethod
    def hard_drop(shape, anchor, board):
        while not TetrisEngine.is_occupied(shape, (anchor[0], anchor[1] + 1), board):
            anchor = (anchor[0], anchor[1] + 1)
        return shape, anchor

    @staticmethod
    def rotate_left(shape, anchor, board):
        new_shape = TetrisEngine.rotated(shape, cclk=False)
        return (shape, anchor) if TetrisEngine.is_occupied(new_shape, anchor, board) else (new_shape, anchor)

    @staticmethod
    def rotate_right(shape, anchor, board):
        new_shape = TetrisEngine.rotated(shape, cclk=True)
        return (shape, anchor) if TetrisEngine.is_occupied(new_shape, anchor, board) else (new_shape, anchor)

    @staticmethod
    def idle(shape, anchor, board):
        return (shape, anchor)

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
