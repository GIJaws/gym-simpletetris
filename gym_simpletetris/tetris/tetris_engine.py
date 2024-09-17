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
        x, y = anchor[0] + i, anchor[1] + j
        if y < 0:
            continue
        if x < 0 or x >= board.shape[0] or y >= board.shape[1] or np.any(board[x, y]):
            return True
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
    while True:
        _, anchor_new = soft_drop(shape, anchor, board)
        if anchor_new == anchor:
            return shape, anchor_new
        anchor = anchor_new


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
        self.board = np.zeros(shape=(width, height, 3), dtype=np.uint8)
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

        # assume that shape, anchor, and board is the same as self
        if self.held_piece is None:
            # If no piece is currently held, hold the current piece
            self.held_piece = self.shape
            self.held_piece_name = self.shape_name
            self._new_piece()  # Generate a new piece to replace the held one
        else:
            # Swap the current piece with the held piece
            self.shape, self.held_piece = self.held_piece, self.shape
            self.shape_name, self.held_piece_name = (
                self.held_piece_name,
                self.shape_name,
            )

        return self.shape, self.anchor

    def _new_piece(self):
        self.anchor = (self.width // 2, 0)
        self.shape_name = self.piece_queue.next_piece()
        self.shape_counts[self.shape_name] += 1
        self.shape = SHAPES[self.shape_name]["shape"]

    def _has_dropped(self):
        return is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board)

    def _clear_lines(self):
        # Check if each row can be cleared (all blocks have non-zero values)
        can_clear = [np.all(np.sum(self.board[:, i], axis=1) > 0) for i in range(self.height)]

        # Create a new board
        new_board = np.zeros_like(self.board)

        # Fill the new board from bottom to top, skipping cleared lines
        j = self.height - 1
        for i in range(self.height - 1, -1, -1):
            if not can_clear[i]:
                new_board[:, j] = self.board[:, i]
                j -= 1

        # Count the number of cleared lines
        cleared_lines = sum(can_clear)
        self.lines_cleared += cleared_lines

        # Update the board
        self.board = new_board

        # Update level if enough lines have been cleared
        if self.lines_cleared >= self.lines_for_next_level:
            self.level += 1
            self.lines_for_next_level += 10  # Increase the threshold for the next level
            self.gravity_interval = self._calculate_gravity_interval()  # Recalculate gravity

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

                    if np.any(self.board[:, 0]):
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
        ghost_anchor = list(self.anchor)
        while not is_occupied(self.shape, (ghost_anchor[0], ghost_anchor[1] + 1), self.board):
            ghost_anchor[1] += 1
        return ghost_anchor

    def render(self):
        self._set_piece(True)
        state = np.copy(self.board)
        self._set_piece(False)

        # Add ghost piece
        ghost_anchor = self.get_ghost_piece_position()
        ghost_color = tuple(max(0, c - 50) for c in SHAPES[self.shape_name]["color"])  # Slightly darker color

        return state, self.shape, ghost_anchor, ghost_color

    def _set_piece(self, on=False):
        if self.shape_name:
            color = SHAPES[self.shape_name]["color"] if on else (0, 0, 0)
            for i, j in self.shape:
                x, y = int(self.anchor[0] + i), int(self.anchor[1] + j)
                if 0 <= x < self.width and 0 <= y < self.height:
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
