from gym_simpletetris.tetris.tetris_shapes import ACTION_NAME_TO_INDEX, SHAPE_NAMES, SHAPES, simplify_board


#  TODO THIS IS HORRIBLE BUT I THINK IT WORKS BE VERY CAREFUL
class FinesseEvaluator:
    def __init__(self, spawn_position, piece_name):
        """
        Initialize the FinesseEvaluator.

        This evaluator tracks the player's actions and determines whether they are making optimal moves (finesse)
        by comparing the actual moves to the minimal required moves to place a piece from its spawn position to its final position.
        """
        self.rotation_actions = {
            ACTION_NAME_TO_INDEX["rotate_left"],
            ACTION_NAME_TO_INDEX["rotate_right"],
        }
        self.translation_actions = {
            ACTION_NAME_TO_INDEX["left"],
            ACTION_NAME_TO_INDEX["right"],
        }
        self.hard_drop_action = ACTION_NAME_TO_INDEX["hard_drop"]
        self.hold_swap_action = ACTION_NAME_TO_INDEX["hold_swap"]
        self.idle_action = ACTION_NAME_TO_INDEX.get("idle", 7)

        self.is_current_finesse = True  # Checks if the current sequence of actions is considered finesse.
        self.finesse_complete = False  # Check if the current sequence of actions completes a finesse move.
        self.reset(piece_name, spawn_position)

    def reset(self, piece_name, spawn_position):
        """
        Resets the state of the finesse evaluator.

        Call this method to start tracking a new sequence of actions for finesse evaluation.
        """
        self.current_actions = []

        self.hold_swap_used = False
        self.last_action_was_hold_swap = False

        # New variables for tracking positions and orientations
        self.piece_name = piece_name
        self.spawn_position = spawn_position
        self.current_position = self.spawn_position
        self.spawn_orientation = 0  # Assuming initial orientation is 0
        self.current_orientation = self.spawn_orientation
        self.cumulative_translation = 0
        self.cumulative_rotation = 0

    def evaluate_action(self, action: int) -> bool:
        """
        Evaluates whether the given action is part of a finesse move.

        Returns True if the action is consistent with finesse, False otherwise.
        """
        # if self.finesse_complete or self.idle_action == action:
        if self.idle_action == action:
            # If the piece is already placed or an idle action, ignore further actions
            return self.is_current_finesse

        # print(f"Evaluating action: {action}")
        self.current_actions.append(action)

        if action == self.hold_swap_action:
            # If hold/swap was not used before and previous action was not hold/swap
            if not self.hold_swap_used and not self.last_action_was_hold_swap:
                self.hold_swap_used = True
                self.is_current_finesse = True
                # Hold/swap is considered finesse under these conditions
            else:
                # Multiple hold/swaps in a row are not finesse
                self.is_current_finesse = False

            self.last_action_was_hold_swap = True

            return self.is_current_finesse
        else:
            self.last_action_was_hold_swap = False

        if action in self.rotation_actions:
            # Update current orientation
            if action == ACTION_NAME_TO_INDEX["rotate_left"]:
                self.current_orientation = (self.current_orientation - 1) % self.get_max_orientations(self.piece_name)
            elif action == ACTION_NAME_TO_INDEX["rotate_right"]:
                self.current_orientation = (self.current_orientation + 1) % self.get_max_orientations(self.piece_name)

            self.cumulative_rotation += 1

        elif action in self.translation_actions:
            # Update current position
            if action == ACTION_NAME_TO_INDEX["left"]:
                self.current_position = (self.current_position[0] - 1, self.current_position[1])
            elif action == ACTION_NAME_TO_INDEX["right"]:
                self.current_position = (self.current_position[0] + 1, self.current_position[1])

            self.cumulative_translation += 1

        elif action == self.hard_drop_action:
            self.finesse_complete = True
        elif action == self.idle_action:
            pass  # Idle actions don't affect finesse
        else:
            # Any other action (e.g., soft drop) is considered in the evaluation
            self.is_current_finesse = False

        self.finesse_complete = action == self.hard_drop_action

        # No need to check thresholds here; evaluation is done after placement
        return self.is_current_finesse

    def piece_placed(self, final_position, final_orientation):
        """
        Called when a piece has been placed.

        Parameters:
            final_position (tuple): The final position (x, y) of the piece.
            final_orientation (int): The final orientation index of the piece.
        """
        self.finesse_complete = True
        self.evaluate_finesse(final_position, final_orientation)

    def evaluate_finesse(self, final_position, final_orientation):
        """
        Evaluates the finesse of the current sequence of actions.

        Sets `self.is_current_finesse` to False if unnecessary moves were made.
        """
        # Calculate minimal required moves
        minimal_translation = abs(final_position[0] - self.spawn_position[0])
        minimal_rotation = self.calculate_minimal_rotation(self.spawn_orientation, final_orientation, self.piece_name)

        # Minimal actions include minimal translations, rotations, and hard drop
        minimal_actions = minimal_translation + minimal_rotation + 1  # +1 for hard drop
        if self.hold_swap_used:
            minimal_actions += 1  # Include hold/swap in minimal actions

        if minimal_rotation == 0:
            minimal_actions -= 1

        # Actual actions taken (excluding idle actions)
        total_actions = len([a for a in self.current_actions if a != self.idle_action])

        self.is_current_finesse = total_actions <= minimal_actions
        print(f"{self.is_current_finesse=}")

        # if not self.is_current_finesse:
        #     breakpoint()

        return self.is_current_finesse

    def calculate_minimal_rotation(self, start_orientation, end_orientation, piece_name):
        """
        Calculates the minimal number of rotations required to get from the start orientation to the end orientation.

        Parameters:
            start_orientation (int): The starting orientation index.
            end_orientation (int): The ending orientation index.
            piece_name (str): The name of the piece.

        Returns:
            int: Minimal number of rotations required.
        """
        max_orientations = self.get_max_orientations(piece_name)
        rotation_diff = (end_orientation - start_orientation) % max_orientations if max_orientations > 1 else 0
        minimal_rotation = min(rotation_diff, max_orientations - rotation_diff)
        return minimal_rotation

    def get_max_orientations(self, piece_name):
        """
        Returns the number of unique orientations for a given piece.

        Parameters:
            piece_name (str): The name of the piece.

        Returns:
            int: Number of unique orientations.
        """
        # O-piece has only 1 orientation; other pieces have 4 or 2
        if piece_name == "O":
            return 1
        elif piece_name in ["I", "S", "Z"]:
            return 2  # These pieces have 2 unique orientations
        else:
            return 4

    def get_finesse_score(self, final_position, final_orientation) -> float:
        """
        Calculate a finesse score for the current sequence of actions.

        Returns:
            float: A score between 0 and 1, where 1 is perfect finesse and 0 is the worst.
        """
        if not self.finesse_complete:
            return 0

        # Recalculate minimal required actions
        minimal_translation = abs(final_position[0] - self.spawn_position[0])
        minimal_rotation = self.calculate_minimal_rotation(self.spawn_orientation, final_orientation, self.piece_name)
        minimal_actions = minimal_translation + minimal_rotation + 1  # +1 for hard drop
        if self.hold_swap_used:
            minimal_actions += 1

        # Actual actions taken (excluding idle actions)
        total_actions = len([a for a in self.current_actions if a != self.idle_action])

        finesse_score = minimal_actions / total_actions if total_actions > 0 else 0.0
        # print(f"Finesse score calculated: {finesse_score}")
        return max(0.0, min(finesse_score, 1.0))
