from .board import TetrisBoard
from .shapes import ShapeFactory


class GameState:
    def __init__(self, width, height):
        self.board = TetrisBoard(width, height)
        self.shape_factory = ShapeFactory()
        self.current_shape, self.current_shape_name = self.shape_factory.get_shape()
        self.anchor = (width // 2, 0)
        self.score = 0
        self.lines_cleared = 0

    def move_left(self):
        new_anchor = (self.anchor[0] - 1, self.anchor[1])
        if self.board.is_valid_position(self.current_shape, new_anchor):
            self.anchor = new_anchor

    def move_right(self):
        new_anchor = (self.anchor[0] + 1, self.anchor[1])
        if self.board.is_valid_position(self.current_shape, new_anchor):
            self.anchor = new_anchor

    def rotate(self):
        new_shape = [(j, -i) for i, j in self.current_shape]
        if self.board.is_valid_position(new_shape, self.anchor):
            self.current_shape = new_shape

    def drop(self):
        while self.board.is_valid_position(
            self.current_shape, (self.anchor[0], self.anchor[1] + 1)
        ):
            self.anchor = (self.anchor[0], self.anchor[1] + 1)

    def lock_piece(self):
        self.board.place_piece(self.current_shape, self.anchor)
        lines_cleared = self.board.clear_lines()
        self.lines_cleared += lines_cleared
        self.score += self._calculate_score(lines_cleared)
        self.current_shape, self.current_shape_name = self.shape_factory.get_shape()
        self.anchor = (self.board.width // 2, 0)

    def _calculate_score(self, lines_cleared):
        return lines_cleared * 100  # Simplified scoring for now

    def is_game_over(self):
        return not self.board.is_valid_position(self.current_shape, self.anchor)
