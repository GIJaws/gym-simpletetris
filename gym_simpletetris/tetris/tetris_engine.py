import random
import numpy as np
from .scoring_system import ScoringSystem

# Adapted from the Tetris engine in the TetrisRL project by jaybutera
# https://github.com/jaybutera/tetrisRL
shapes = {
    "T": [(0, 0), (-1, 0), (1, 0), (0, -1)],
    "J": [(0, 0), (-1, 0), (0, -1), (0, -2)],
    "L": [(0, 0), (1, 0), (0, -1), (0, -2)],
    "Z": [(0, 0), (-1, 0), (0, -1), (1, -1)],
    "S": [(0, 0), (-1, -1), (0, -1), (1, 0)],
    "I": [(0, 0), (0, -1), (0, -2), (0, -3)],
    "O": [(0, 0), (0, -1), (-1, 0), (-1, -1)],
}
shape_names = ["T", "J", "L", "Z", "S", "I", "O"]


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
        if x < 0 or x >= board.shape[0] or y >= board.shape[1] or board[x, y]:
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
    ):
        self.width = width
        self.height = height
        self.board = np.zeros(shape=(width, height), dtype=np.float32)
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
            0: left,
            1: right,
            2: hard_drop,
            3: soft_drop,
            4: rotate_left,
            5: rotate_right,
            6: self.hold_swap,
            7: idle,
            8: lambda shape, anchor, board: rotate_left(*left(shape, anchor, board), board),
            9: lambda shape, anchor, board: rotate_right(*right(shape, anchor, board), board),
        }
        self.action_value_map = dict([(j, i) for i, j in self.value_action_map.items()])
        self.nb_actions = len(self.value_action_map)

        self.held_piece = None  # No piece is held at the start
        self.next_piece = None

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
        self.shape = None
        self.shape_name = None
        self.n_deaths = 0

        self._lock_delay_fn = lambda x: (x + 1) % (max(lock_delay, 0) + 1)
        self._lock_delay = 0
        self._step_reset = step_reset

        self.shape_counts = dict(zip(shape_names, [0] * len(shapes)))

    def _choose_shape(self):
        values = list(self.shape_counts.values())
        maxm = max(values)
        m = [5 + maxm - x for x in values]
        r = random.randint(1, sum(m))
        for i, n in enumerate(m):
            r -= n
            if r <= 0:
                return shape_names[i]

    def hold_swap(self, shape, anchor, board):
        if self.held_piece is None:
            # If no piece is currently held, hold the current piece
            self.held_piece = shape
            self._new_piece()  # Generate a new piece to replace the held one
        else:
            # Swap the current piece with the held piece
            shape, self.held_piece = self.held_piece, shape

        return shape, anchor

    def _new_piece(self):
        self.anchor = (self.width / 2, 0)
        self.shape_name = self._choose_shape()
        self.shape_counts[self.shape_name] += 1
        self.shape = shapes[self.shape_name]

    def _has_dropped(self):
        return is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board)

    def _clear_lines(self):
        can_clear = [np.all(self.board[:, i]) for i in range(self.height)]
        new_board = np.zeros_like(self.board)
        j = self.height - 1
        for i in range(self.height - 1, -1, -1):
            if not can_clear[i]:
                new_board[:, j] = self.board[:, i]
                j -= 1
        cleared_lines = sum(can_clear)
        self.lines_cleared += cleared_lines
        self.board = new_board

        # Update level
        if self.lines_cleared >= self.lines_for_next_level:
            self.level += 1
            self.lines_for_next_level += 10  # Increase lines needed for next level
            self.gravity_interval = self._calculate_gravity_interval()  # Recalculate speed

        return cleared_lines

    def _count_holes(self):
        self.holes = np.count_nonzero(self.board.cumsum(axis=1) * ~self.board.astype(bool))
        return self.holes

    def valid_action_count(self):
        valid_action_sum = 0

        for value, fn in self.value_action_map.items():
            if fn(self.shape, self.anchor, self.board) != (self.shape, self.anchor):
                valid_action_sum += 1

        return valid_action_sum

    def get_info(self):
        return {
            "time": self.time,
            "current_piece": self.shape_name,
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "holes": self.holes,
            "deaths": self.n_deaths,
            "statistics": self.shape_counts,
            "level": self.level,
            "gravity_interval": self.gravity_interval,
        }

    def _calculate_gravity_interval(self):
        # Level 0: gravity is effectively disabled, requiring player input for dropping
        if self.level == 0:
            return float("inf")  # Gravity won't trigger automatically

        # Other levels: use the regular gravity decay formula
        return max(
            1,
            min(
                60 * (0.8 - ((min(self.level + 9, 29) - 1) * 0.007)) ** min(self.level + 9, 29),
                60,
            ),
        )

    def step(self, action):
        self.anchor = (int(self.anchor[0]), int(self.anchor[1]))
        self.shape, self.anchor = self.value_action_map[action](self.shape, self.anchor, self.board)

        self.time += 1
        self.gravity_timer += 1
        reward = self.scoring_system.calculate_step_reward()

        done = False
        cleared_lines = 0

        if action == 2 or self.gravity_timer >= self.gravity_interval and self.gravity_interval != float("inf"):
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
        return state, reward, done, cleared_lines

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

    def render(self):
        self._set_piece(True)
        state = np.copy(self.board)
        self._set_piece(False)
        return state

    def _set_piece(self, on=False):
        for i, j in self.shape:
            x, y = i + self.anchor[0], j + self.anchor[1]
            if x < self.width and x >= 0 and y < self.height and y >= 0:
                self.board[int(self.anchor[0] + i), int(self.anchor[1] + j)] = on

    def __repr__(self):
        self._set_piece(True)
        s = "o" + "-" * self.width + "o\n"
        s += "\n".join(["|" + "".join(["X" if j else " " for j in i]) + "|" for i in self.board.T])
        s += "\no" + "-" * self.width + "o"
        self._set_piece(False)
        return s
