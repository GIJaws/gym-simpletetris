import numpy as np

# Adapted from the Tetris engine in the TetrisRL project by jaybutera
# https://github.com/jaybutera/tetrisRL
SHAPES = {  # RGB here is GBR
    "T": {"shape": ((0, 0), (-1, 0), (1, 0), (0, -1)), "color": (128, 0, 128)},  # Purple
    "J": {"shape": ((0, 0), (-1, 0), (0, -1), (0, -2)), "color": (0, 255, 0)},  # Blue
    "L": {"shape": ((0, 0), (1, 0), (0, -1), (0, -2)), "color": (165, 0, 255)},  # Orange
    "Z": {"shape": ((0, 0), (-1, 0), (0, -1), (1, -1)), "color": (0, 0, 255)},  # Red
    "S": {"shape": ((0, 0), (-1, -1), (0, -1), (1, 0)), "color": (255, 0, 0)},  # Green
    "I": {"shape": ((0, 0), (0, -1), (0, -2), (0, -3)), "color": (255, 255, 0)},  # Cyan
    "O": {"shape": ((0, 0), (0, -1), (-1, 0), (-1, -1)), "color": (0, 255, 255)},  # Yellow
}
SHAPE_NAMES = tuple(SHAPES.keys())
BASIC_ACTIONS = {
    0: "left",  # Move Left
    1: "right",  # Move Right
    2: "rotate_left",  # Rotate Left
    3: "rotate_right",  # Rotate Right
    4: "hold_swap",  # Hold/Swap
    5: "hard_drop",  # Hard Drop
    6: "soft_drop",  # Soft Drop
    # 7: "idle",  # Idle
}  # TODO for now lets not worry about soft dropping or idling until we get a model that can actually sorta play

# List of valid action combinations based on Tetris engine logic
# ? Can have more then one action as its a list of actions but for now lets just get
#  ? the model clearing a line consistently
ACTION_COMBINATIONS = {ii: [ii] for ii in BASIC_ACTIONS}


def bitmask_to_actions(action_bitmask):
    """
    Converts a bitmask integer to a list of basic action indices.
    Args:
        action_bitmask (int): Bitmask representing the action combination.
    Returns:
        list: List of action indices corresponding to the bitmask.
    """
    actions = []
    for action, name in BASIC_ACTIONS.items():
        if action_bitmask & (1 << action):
            actions.append(action)
    return actions


def simplify_board(board):
    """
    Simplifies the board representation by converting it to a 2D binary array.

    Args:
        board (np.ndarray): The original board with shape (width, height, channels).

    Returns:
        np.ndarray: Simplified board with shape (width, height) as float32.
    """
    # TODO can I change this to bool or int? would that be better?
    if board.ndim == 3:
        return np.any(board != 0, axis=2).astype(np.float32)
    elif board.ndim == 2:
        return board.astype(np.float32)
    else:
        raise ValueError("Invalid board shape. Expected 2D or 3D array.")


# Reverse mapping: action name to index
ACTION_NAME_TO_INDEX = {name: action for action, name in BASIC_ACTIONS.items()}

# TODO OTHER CONSTANTS I NEED TO MOVE TO A NEW FILE
WIDTH = 10  # DEFAULT WIDTH OF THE PLAY FIELD
HEIGHT = 20  # DEFAULT HEIGHT OF THE PLAY FIELD THE USER CAN SEE
BUFFER_HEIGHT = 20  # DEFAULT BUFFER HEIGHT FOR THE PLAY FIELD

# If the hardware permits, a sliver of the 21st row is shown to aid players manipulate the active
#  piece in that area.
VISIBLE_HEIGHT = 21
