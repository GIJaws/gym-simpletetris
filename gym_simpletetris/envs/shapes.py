import random

SHAPES = {
    "T": [(0, 0), (-1, 0), (1, 0), (0, -1)],
    "J": [(0, 0), (-1, 0), (0, -1), (0, -2)],
    "L": [(0, 0), (1, 0), (0, -1), (0, -2)],
    "Z": [(0, 0), (-1, 0), (0, -1), (1, -1)],
    "S": [(0, 0), (-1, -1), (0, -1), (1, 0)],
    "I": [(0, 0), (0, -1), (0, -2), (0, -3)],
    "O": [(0, 0), (0, -1), (-1, 0), (-1, -1)],
}


class ShapeFactory:
    def __init__(self):
        self.shape_counts = {shape: 0 for shape in SHAPES.keys()}

    def get_shape(self):
        shape_name = self._choose_shape()
        self.shape_counts[shape_name] += 1
        return SHAPES[shape_name], shape_name

    def _choose_shape(self):
        max_count = max(self.shape_counts.values())
        weights = [max_count + 5 - count for count in self.shape_counts.values()]
        return random.choices(list(SHAPES.keys()), weights=weights)[0]
