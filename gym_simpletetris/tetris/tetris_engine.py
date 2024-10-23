from typing import Literal
import numpy as np
from pygame.draw import lines

# from gym_simpletetris.tetris.finesse_evaluator import FinesseEvaluator
from gym_simpletetris.tetris.pieces import Piece
from gym_simpletetris.tetris.board import Board
import math

from dataclasses import dataclass, replace, field


from abc import ABC, abstractmethod


class Action(ABC):
    priority = 0

    @staticmethod
    @abstractmethod
    def apply(state: "GameState") -> "GameState":
        """Apply the action to the game state and return a new game state."""
        from gym_simpletetris.tetris.tetris_engine import GameState

        pass

    @staticmethod
    def rotate(piece: Piece, direction: Literal["left", "right"], board: Board) -> Piece:
        if direction == "left":
            rotated_piece = piece.rotate(clockwise=False)
        elif direction == "right":
            rotated_piece = piece.rotate(clockwise=True)
        else:
            raise ValueError("Invalid rotation direction (btw this should never be called, what did you do?)")

        wall_kick_offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

        pos_x = rotated_piece.position[0]
        pos_y = rotated_piece.position[1]

        for dx, dy in wall_kick_offsets:
            new_position = (pos_x + dx, pos_y + dy)

            if not board.collision(new_piece := replace(rotated_piece, position=new_position)):
                return new_piece
        return piece

    @staticmethod
    def move(piece: Piece, dx: int, dy: int) -> Piece:
        new_x = max(0, int(piece.position[0]) + dx)  # Ensure x doesn't go below 0
        new_y = max(0, int(piece.position[1]) + dy)  # Ensure y doesn't go below 0
        return replace(piece, position=(new_x, new_y))


@dataclass(frozen=True)
class GameState:
    # ? GameState does not handle lives or deaths, they are handled in the environment
    board: Board  # Assumption that the board stored in GameState does not include the current piece that is falling
    current_piece: Piece
    next_pieces: tuple[Piece, ...]
    held_piece: Piece | None = None
    step_score: int = 0
    score: int = 0
    step_lines_cleared: int = 0
    lines_cleared: int = 0
    level: int = 1
    game_over: bool = False  # TODO is it safe to default GameOver to false?
    hold_used: bool = False
    lock_delay_counter: int = 0
    MAX_LOCK_DELAY: int = 10  # 100  # TODO IDK WHAT TO PUT HERE, 0 means as soon as it touches it is placed
    current_time: int = 0
    gravity_interval: float = 0
    gravity_timer: int = 0
    piece_timer: int = 0
    info: dict = field(default_factory=dict)

    def _calculate_gravity_interval(self) -> int:
        if self.level == 0:
            return 0
        base_interval = 60  # Starting interval for level 1
        difficulty_factor = 0.2  # Adjust this to control difficulty progression
        return max(1, int(base_interval * math.exp(-difficulty_factor * (self.level - 1))))

    def step(self, actions: list[type[Action]]) -> "GameState":

        state = replace(self, step_lines_cleared=0, step_score=0).apply_actions(actions)
        if not state.game_over:
            state = state.process_game_step()

        return replace(state, info=state._generate_info())

    def apply_actions(self, actions: list[type[Action]]) -> "GameState":
        state = self
        for action in sorted(actions, key=lambda a: a.priority):
            if state.game_over:
                break
            state = action.apply(state)
        return replace(
            state,
            current_time=state.current_time + 1,
            gravity_timer=state.gravity_timer + 1,
            piece_timer=state.piece_timer + 1,
        )

    def process_game_step(self) -> "GameState":
        state = self
        current_piece = state.current_piece
        lock_delay_counter = state.lock_delay_counter
        gravity_timer = 0
        if not state.board.collision(state.current_piece):
            if (state.gravity_timer >= state.gravity_interval) and state.gravity_interval:
                if not state.board.collision(new_piece := Action.move(state.current_piece, dx=0, dy=1)):
                    current_piece = new_piece
                    lock_delay_counter = 0
            else:
                gravity_timer = state.gravity_timer + 1

        state = replace(
            state, gravity_timer=gravity_timer, current_piece=current_piece, lock_delay_counter=lock_delay_counter
        )
        lock_delay_counter = state.lock_delay_counter

        if state.board.collision(state.current_piece) and state.MAX_LOCK_DELAY:
            lock_delay_counter += 1

        state = replace(state, lock_delay_counter=lock_delay_counter)

        if state.lock_delay_counter < state.MAX_LOCK_DELAY or not state.MAX_LOCK_DELAY:
            return state

        return state.place_current_piece()

    def calculate_score(self) -> "GameState":
        """
        Update the score and level based on lines cleared.
        """
        scoring = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        new_step_score = (
            self.step_score + scoring[self.step_lines_cleared]
        )  # TODO lets see if this ever results in an error
        new_level = self.level
        new_gravity_interval = self.gravity_interval

        if self.lines_cleared // 10 > self.level - 1:
            new_level += 1
            new_gravity_interval = self._calculate_gravity_interval()

        return replace(
            self,
            step_score=new_step_score,
            score=self.score + new_step_score,
            level=new_level,
            gravity_interval=new_gravity_interval,
        )

    def place_current_piece(self) -> "GameState":
        # Method to place the current piece on the board and clear lines and then returns the updated GameState

        print(f"Placing current piece {self.current_piece.name} at {self.current_piece.position}")
        board, step_lines_cleared = self.board.place_piece(self.current_piece).clear_lines()
        piece, *next_pieces = self.next_pieces
        piece = board.set_piece_spawn_position(piece)
        state = replace(
            self,
            current_piece=piece,
            next_pieces=next_pieces,
            board=board,
            hold_used=False,
            lock_delay_counter=0,
            piece_timer=0,
            game_over=board.collision(piece),
            # ? Assume step_lines_clear is reset to 0 at the start of each time step
            step_lines_cleared=self.step_lines_cleared + step_lines_cleared,
            lines_cleared=self.lines_cleared + step_lines_cleared,
        )

        return state.calculate_score()

    def _generate_info(self) -> dict:
        return {
            "score": self.step_score,
            "level": self.level,
            "lines_cleared": self.lines_cleared,
            "current_piece": self.current_piece.name,
            "next_piece": self.next_pieces[0].name if self.next_pieces else None,
            "held_piece": self.held_piece.name if self.held_piece else None,
            # Add any other relevant information
        }

    def get_ghost_piece(self) -> Piece:
        ghost_piece = self.current_piece
        while not self.board.collision(_ghost_piece := Action.move(ghost_piece, dx=0, dy=1)):
            ghost_piece = _ghost_piece
        return ghost_piece

    @staticmethod
    def create_initial_game_state(
        width: int,
        height: int,
        buffer_height: int,
        current_piece: Piece,
        next_pieces: tuple[Piece, ...],
        initial_level: int,
        held_piece: Piece | None = None,
    ) -> "GameState":
        board = Board.create_board(width=width, height=height, buffer_height=buffer_height)
        init_state = GameState(
            board=board,
            current_piece=board.set_piece_spawn_position(current_piece),
            next_pieces=next_pieces,
            held_piece=held_piece,
            step_score=0,
            level=initial_level,
        )
        init_state = replace(init_state, gravity_interval=init_state._calculate_gravity_interval())
        return replace(init_state, info=init_state._generate_info())
