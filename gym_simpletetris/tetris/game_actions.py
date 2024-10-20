from gym_simpletetris.tetris.tetris_engine import Action, GameState


from dataclasses import replace

from enum import Enum


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
        while not state.board.collision(_piece := Action.move(piece, dx=0, dy=1)):
            piece = _piece

        return replace(state, current_piece=piece).place_current_piece()


class Hold(Action):

    priority = 5

    @staticmethod
    def apply(state: GameState) -> GameState:
        if state.hold_used:
            return state
        if state.held_piece:
            return replace(
                state,
                current_piece=state.board.set_piece_spawn_position(state.held_piece),
                held_piece=state.current_piece,
                hold_used=True,
            )
        else:
            piece, *remaining_pieces = state.next_pieces
            return replace(
                state,
                current_piece=state.board.set_piece_spawn_position(piece),
                next_pieces=remaining_pieces,
                held_piece=state.current_piece,
                hold_used=True,
            )


class Idle(Action):

    @staticmethod
    def apply(state: GameState) -> GameState:
        return state  # Do nothing


class GameAction(Enum):
    MOVE_LEFT = (MoveLeft, 0)
    MOVE_RIGHT = (MoveRight, 1)
    ROTATE_LEFT = (RotateLeft, 2)
    ROTATE_RIGHT = (RotateRight, 3)
    HARD_DROP = (HardDrop, 4)
    SOFT_DROP = (SoftDrop, 5)
    HOLD = (Hold, 6)
    IDLE = (Idle, 7)

    def __init__(self, action_class: type[Action], index: int):
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
    def from_index(cls, *indices: int) -> list[type["Action"]]:
        # ! Assume indices are valid
        return [action.action_class for action in cls if action.index in indices]

    @cached_property
    @classmethod
    def get_combinations(cls):
        return {action.index: [action.index] for action in cls}
