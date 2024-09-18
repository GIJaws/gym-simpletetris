import random
import numpy as np
from .scoring_system import ScoringSystem
from .tetris_shapes import SHAPE_NAMES, SHAPES
import math


def rotated(shape, cclk=False):
    if cclk:
        return [(-j, i) for i, j in shape]
    else:
        return [(j, -i) for i, j in shape]


def is_occupied(shape, anchor, board):
    for i, j in shape:
        x = anchor[0] + i
        y = anchor[1] + j
        if x < 0 or x >= board.shape[0] or y >= board.shape[1]:
            return True  # Out of bounds
        if np.any(board[x, y]):
            return True  # Position already occupied
    return False


def left(shape, anchor, board):
    new_anchor = (anchor[0] - 1, anchor[1])
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def right(shape, anchor, board):
    new_anchor = (anchor[0] + 1, anchor[1])

    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def soft_drop(shape, anchor, board):
    new_anchor = (anchor[0], anchor[1] + 1)
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def hard_drop(shape, anchor, board):
    while not is_occupied(shape, (anchor[0], anchor[1] + 1), board):
        anchor = (anchor[0], anchor[1] + 1)
    return shape, anchor


def rotate_left(shape, anchor, board):
    new_shape = rotated(shape, cclk=False)
    return (shape, anchor) if is_occupied(new_shape, anchor, board) else (new_shape, anchor)


def rotate_right(shape, anchor, board):
    new_shape = rotated(shape, cclk=True)
    return (shape, anchor) if is_occupied(new_shape, anchor, board) else (new_shape, anchor)


def idle(shape, anchor, board):
    return (shape, anchor)


class TetrisEngine:
    def __init__(
        self,
        width,
        height,
        buffer_height,
        lock_delay=0,
        step_reset=False,
        reward_step=False,
        penalise_height=False,
        penalise_height_increase=False,
        advanced_clears=False,
        high_scoring=False,
        penalise_holes=False,
        penalise_holes_increase=False,
        initial_level=1,
        preview_size=4,
    ):
        self.width = width
        self.height = height
        self.buffer_height = buffer_height
        self.total_height = self.height + self.buffer_height
        # Initialize the board with buffer zone at the bottom
        self.board = np.zeros(shape=(width, self.total_height, 3), dtype=np.uint8)

        self.scoring_system = ScoringSystem(
            reward_step,
            penalise_height,
            penalise_height_increase,
            advanced_clears,
            high_scoring,
            penalise_holes,
            penalise_holes_increase,
        )

        self.value_action_map = {
            0: left,  # Move Left
            1: right,  # Move Right
            2: hard_drop,  # Hard Drop
            3: soft_drop,  # Soft Drop
            4: rotate_left,  # Rotate Left
            5: rotate_right,  # Rotate Right
            6: self.hold_swap,  # Hold/Swap
            7: idle,  # Idle
        }
        # TODO REMOVE THESE COMMENTS AND WORK OUT WHAT ELSE IS NOT BEING USED
        # self.action_value_map = dict([(j, i) for i, j in self.value_action_map.items()])
        # self.nb_actions = len(self.value_action_map)

        self.held_piece = None  # No piece is held at the start
        self.held_piece_name = None
        self.hold_used = False
        self.piece_queue = PieceQueue(preview_size)
        self.shape_counts = dict(zip(SHAPE_NAMES, [0] * len(SHAPES)))
        self.shape = None
        self.shape_name = None
        self._new_piece()

        self.initial_level = initial_level
        self.level = initial_level
        self.lines_for_next_level = self.level * 10  # Number of lines to clear for next level
        self.gravity_interval = self._calculate_gravity_interval()
        self.gravity_timer = 0
        self.gravity_counter = 0

        self.time = -1
        self.score = -1
        self.holes = 0
        self.lines_cleared = 0
        self.piece_height = 0
        self.anchor = None

        self.n_deaths = 0

        self._lock_delay_fn = lambda x: (x + 1) % (max(lock_delay, 0) + 1)
        self._lock_delay = 0
        self._step_reset = step_reset

    # TODO make this functional so no side effects can occur to prevent future bugs
    def hold_swap(self, shape, anchor, board):
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
        return is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board)

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
        self.holes = np.count_nonzero(self.board.cumsum(axis=1) * ~self.board.astype(bool))
        return self.holes

    def get_info(self):
        info = {
            "time": self.time,
            "current_piece": self.shape_name,
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "holes": self.holes,
            "deaths": self.n_deaths,
            "statistics": self.shape_counts,
            "level": self.level,
            "gravity_interval": self.gravity_interval,
            "next_piece": self.piece_queue.get_preview(),
            "held_piece": self.held_piece,
            "held_piece_name": self.held_piece_name,
        }

        # print(info)

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
            2: 3,  # Hard Drop (highest priority)
            3: 2,  # Soft Drop
            4: 1,  # Rotate Left
            5: 1,  # Rotate Right
            6: 2,  # Hold/Swap
            7: 0,  # Idle (lowest priority)
        }

        # Process each action in the sorted order
        for action in sorted(actions, key=lambda action: action_priority[action]):
            self.anchor = (int(self.anchor[0]), int(self.anchor[1]))
            self.shape, self.anchor = self.value_action_map[action](self.shape, self.anchor, self.board)

        self.time += 1
        self.gravity_timer += 1
        reward = self.scoring_system.calculate_step_reward()

        done = False
        cleared_lines = 0

        if 2 in actions or self.gravity_timer >= self.gravity_interval and self.gravity_interval != float("inf"):
            self.gravity_timer = 0
            self.shape, new_anchor = soft_drop(self.shape, self.anchor, self.board)
            if self._step_reset and (self.anchor != new_anchor):
                self._lock_delay = 0

            self.anchor = new_anchor

            if self._has_dropped():
                self._lock_delay = self._lock_delay_fn(self._lock_delay)

                if self._lock_delay == 0:
                    self._set_piece(True)
                    cleared_lines = self._clear_lines()

                    reward += self.scoring_system.calculate_clear_reward(cleared_lines)
                    self.score += self.scoring_system.calculate_clear_reward(cleared_lines)

                    if np.any(self.board[:, : self.buffer_height]):
                        self._count_holes()
                        self.n_deaths += 1
                        done = True
                        reward = -100
                    else:
                        old_holes = self.holes
                        old_height = self.piece_height
                        self._count_holes()
                        new_height = sum(np.any(self.board, axis=0))

                        reward += self.scoring_system.calculate_height_penalty(self.board)
                        reward += self.scoring_system.calculate_height_increase_penalty(new_height, old_height)
                        reward += self.scoring_system.calculate_holes_penalty(self.holes)
                        reward += self.scoring_system.calculate_holes_increase_penalty(self.holes, old_holes)

                        self.piece_height = new_height
                        self._new_piece()
                        if is_occupied(self.shape, self.anchor, self.board):
                            self.n_deaths += 1
                            done = True
                            reward = -100

        self._set_piece(True)
        state = np.copy(self.board)
        self._set_piece(False)
        return state, reward, done

    def clear(self):
        self.time = 0
        self.score = 0
        self.holes = 0
        self.lines_cleared = 0
        self.piece_height = 0
        self._new_piece()
        self.board = np.zeros_like(self.board)

        self.level = self.initial_level
        self.lines_for_next_level = 10
        self.gravity_interval = self._calculate_gravity_interval()
        self.gravity_timer = 0

        return self.board

    def get_ghost_piece_position(self):
        ghost_anchor = self.anchor
        while not is_occupied(self.shape, ghost_anchor, self.board):
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
        if self.shape_name:
            color = SHAPES[self.shape_name]["color"] if on else (0, 0, 0)
            for i, j in self.shape:
                x, y = int(self.anchor[0] + i), int(self.anchor[1] + j)
                # TODO should the '0' in `0 <= y < self.total_height` be self.buffer_height??????
                if 0 <= x < self.width and 0 <= y < self.total_height:
                    self.board[x, y] = color

    def __repr__(self):
        self._set_piece(True)
        s = "o" + "-" * self.width + "o\n"
        s += "\n".join(["|" + "".join(["X" if j else " " for j in i]) + "|" for i in self.board.T])
        s += "\no" + "-" * self.width + "o"
        self._set_piece(False)
        return s


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
