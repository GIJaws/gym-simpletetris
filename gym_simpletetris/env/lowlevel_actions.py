from __future__ import annotations

from gym_simpletetris.core.game_actions import GameAction

ACTION_SPACE_ID = "lowlevel_12"

_LL_MOVE_LEFT = GameAction.MOVE_LEFT.index
_LL_MOVE_RIGHT = GameAction.MOVE_RIGHT.index
_LL_ROTATE_LEFT = GameAction.ROTATE_LEFT.index
_LL_ROTATE_RIGHT = GameAction.ROTATE_RIGHT.index
_LL_HARD_DROP = GameAction.HARD_DROP.index
_LL_HOLD = GameAction.HOLD.index

LOWLEVEL_ACTION_MAP: tuple[tuple[int, ...], ...] = (
    (_LL_MOVE_LEFT,),
    (_LL_MOVE_RIGHT,),
    (_LL_ROTATE_RIGHT,),
    (_LL_ROTATE_LEFT,),
    (_LL_HOLD,),
    (_LL_HARD_DROP,),
    (_LL_MOVE_LEFT, _LL_ROTATE_RIGHT),
    (_LL_MOVE_RIGHT, _LL_ROTATE_RIGHT),
    (_LL_MOVE_LEFT, _LL_ROTATE_LEFT),
    (_LL_MOVE_RIGHT, _LL_ROTATE_LEFT),
    (_LL_MOVE_LEFT, _LL_HARD_DROP),
    (_LL_MOVE_RIGHT, _LL_HARD_DROP),
)

ACTION_NAMES: tuple[str, ...] = (
    "left",
    "right",
    "rot_cw",
    "rot_ccw",
    "hold",
    "drop",
    "L+rot_cw",
    "R+rot_cw",
    "L+rot_ccw",
    "R+rot_ccw",
    "L+drop",
    "R+drop",
)

N_LOWLEVEL_ACTIONS = len(LOWLEVEL_ACTION_MAP)


def lowlevel_action_tuple(action: int) -> tuple[int, ...]:
    action_i = int(action)
    if not 0 <= action_i < N_LOWLEVEL_ACTIONS:
        raise ValueError(f"action {action_i} out of range [0, {N_LOWLEVEL_ACTIONS})")
    return LOWLEVEL_ACTION_MAP[action_i]


def lowlevel_action_name(action: int) -> str:
    action_i = int(action)
    if not 0 <= action_i < N_LOWLEVEL_ACTIONS:
        raise ValueError(f"action {action_i} out of range [0, {N_LOWLEVEL_ACTIONS})")
    return ACTION_NAMES[action_i]
