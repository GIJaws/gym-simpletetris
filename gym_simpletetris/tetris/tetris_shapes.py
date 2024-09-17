# Adapted from the Tetris engine in the TetrisRL project by jaybutera
# https://github.com/jaybutera/tetrisRL
SHAPES = {
    "T": {"shape": ((0, 0), (-1, 0), (1, 0), (0, -1)), "color": (128, 0, 128)},  # Purple
    "J": {"shape": ((0, 0), (-1, 0), (0, -1), (0, -2)), "color": (0, 0, 255)},  # Blue
    "L": {"shape": ((0, 0), (1, 0), (0, -1), (0, -2)), "color": (255, 165, 0)},  # Orange
    "Z": {"shape": ((0, 0), (-1, 0), (0, -1), (1, -1)), "color": (255, 0, 0)},  # Red
    "S": {"shape": ((0, 0), (-1, -1), (0, -1), (1, 0)), "color": (0, 255, 0)},  # Green
    "I": {"shape": ((0, 0), (0, -1), (0, -2), (0, -3)), "color": (0, 255, 255)},  # Cyan
    "O": {"shape": ((0, 0), (0, -1), (-1, 0), (-1, -1)), "color": (255, 255, 0)},  # Yellow
}
SHAPE_NAMES = ("T", "J", "L", "Z", "S", "I", "O")
