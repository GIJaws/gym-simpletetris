# Adapted from the Tetris engine in the TetrisRL project by jaybutera
# https://github.com/jaybutera/tetrisRL
SHAPES = {
    "T": [(0, 0), (-1, 0), (1, 0), (0, -1)],
    "J": [(0, 0), (-1, 0), (0, -1), (0, -2)],
    "L": [(0, 0), (1, 0), (0, -1), (0, -2)],
    "Z": [(0, 0), (-1, 0), (0, -1), (1, -1)],
    "S": [(0, 0), (-1, -1), (0, -1), (1, 0)],
    "I": [(0, 0), (0, -1), (0, -2), (0, -3)],
    "O": [(0, 0), (0, -1), (-1, 0), (-1, -1)],
}
SHAPE_NAMES = ["T", "J", "L", "Z", "S", "I", "O"]
