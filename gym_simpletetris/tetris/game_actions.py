from gym_simpletetris.tetris.tetris_engine import Action, GameState


from dataclasses import replace

from enum import Enum, auto


from functools import cached_property


class MoveLeft(Action):
    priority = 1

    @staticmethod
    def apply(state: GameState) -> GameState:
        piece = Action.move(state.current_piece, dx=-1, dy=0)
        if state.board.collision(piece):
            # Undo move by not changing the piece
            return state
        return replace(state, **{"current_piece": piece})


class MoveRight(Action):
    priority = 1

    @staticmethod
    def apply(state: GameState) -> GameState:
        piece = Action.move(state.current_piece, dx=1, dy=0)
        if state.board.collision(piece):
            # Undo move by not changing the piece
            return state
        return replace(state, **{"current_piece": piece})


class RotateLeft(Action):
    priority = 1

    @staticmethod
    def apply(state: GameState) -> GameState:
        piece = Action.rotate(state.current_piece, direction="left", board=state.board)
        if state.board.collision(piece):
            # Undo move by not changing the piece
            return state
        return replace(state, **{"current_piece": piece})


class RotateRight(Action):
    priority = 1

    @staticmethod
    def apply(state: GameState) -> GameState:
        piece = Action.rotate(state.current_piece, direction="right", board=state.board)
        if state.board.collision(piece):
            # Undo move by not changing the piece
            return state
        return replace(state, **{"current_piece": piece})


class SoftDrop(Action):
    priority = 2

    @staticmethod
    def apply(state: GameState) -> GameState:
        piece = Action.move(state.current_piece, dx=0, dy=1)
        if state.board.collision(piece):
            # Undo move by not changing the piece
            return state
        return replace(state, **{"current_piece": piece})


class HardDrop(Action):
    priority = 4

    @staticmethod
    def apply(state: GameState) -> GameState:
        piece = state.current_piece
        while not state.board.collision(piece := Action.move(piece, dx=0, dy=1)):
            continue

        board, lines_cleared = state.board.place_piece(piece).clear_lines()

        state = replace(state, board=board).calculate_score(lines_cleared)

        piece, *next_pieces = state.next_pieces

        return replace(
            state,
            current_piece=board.set_spawn_position(piece),
            next_pieces=next_pieces,
            hold_used=False,
            lock_delay_counter=0,
            game_over=board.collision(piece),
        )


class Hold(Action):

    priority = 5

    @staticmethod
    def apply(state: GameState) -> GameState:
        if state.hold_used:
            return state
        if state.held_piece:
            return replace(
                state,
                current_piece=state.board.set_spawn_position(state.held_piece),
                held_piece=state.current_piece,
                hold_used=True,
            )
        else:
            piece, *remaining_pieces = state.next_pieces
            return replace(
                state,
                current_piece=state.board.set_spawn_position(piece),
                next_pieces=remaining_pieces,
                held_piece=state.current_piece,
                hold_used=True,
            )


class Idle(Action):

    @staticmethod
    def apply(state: GameState) -> GameState:
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
