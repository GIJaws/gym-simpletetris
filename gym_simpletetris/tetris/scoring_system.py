class ScoringSystem:
    def __init__(
        self,
        reward_step=False,
        penalise_height=False,
        penalise_height_increase=False,
        advanced_clears=False,
        high_scoring=False,
        penalise_holes=False,
        penalise_holes_increase=False,
    ):
        self.reward_step = reward_step
        self.penalise_height = penalise_height
        self.penalise_height_increase = penalise_height_increase
        self.advanced_clears = advanced_clears
        self.high_scoring = high_scoring
        self.penalise_holes = penalise_holes
        self.penalise_holes_increase = penalise_holes_increase

    def calculate_step_reward(self):
        return 1 if self.reward_step else 0

    def calculate_clear_reward(self, cleared_lines):
        if self.advanced_clears:
            scores = [0, 40, 100, 300, 1200]
            return 2.5 * scores[cleared_lines]
        elif self.high_scoring:
            return 1000 * cleared_lines
        else:
            return 100 * cleared_lines

    def calculate_height_penalty(self, board):
        if self.penalise_height:
            return -sum(any(board[:, i]) for i in range(board.shape[1]))
        return 0

    def calculate_height_increase_penalty(self, new_height, old_height):
        if self.penalise_height_increase and new_height > old_height:
            return -10 * (new_height - old_height)
        return 0

    def calculate_holes_penalty(self, holes):
        if self.penalise_holes:
            return -5 * holes
        return 0

    def calculate_holes_increase_penalty(self, new_holes, old_holes):
        if self.penalise_holes_increase:
            return -5 * (new_holes - old_holes)
        return 0
