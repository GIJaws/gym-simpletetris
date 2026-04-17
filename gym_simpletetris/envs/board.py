import numpy as np


class TetrisBoard:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height), dtype=np.float32)

    def is_valid_position(self, shape, anchor):
        for i, j in shape:
            x, y = anchor[0] + i, anchor[1] + j
            if (
                x < 0
                or x >= self.width
                or y >= self.height
                or (y >= 0 and self.grid[x, y])
            ):
                return False
        return True

    def place_piece(self, shape, anchor):
        for i, j in shape:
            x, y = anchor[0] + i, anchor[1] + j
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[x, y] = 1

    def clear_lines(self):
        lines_cleared = 0
        for y in range(self.height - 1, -1, -1):
            if np.all(self.grid[:, y]):
                self.grid[:, y + 1 :] = self.grid[:, y:-1]
                self.grid[:, 0] = 0
                lines_cleared += 1
        return lines_cleared

    def count_holes(self):
        return np.count_nonzero(self.grid.cumsum(axis=1) * ~self.grid.astype(bool))
