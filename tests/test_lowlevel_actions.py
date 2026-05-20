import pytest

from gym_simpletetris.core.game_actions import GameAction
from gym_simpletetris.env.lowlevel_actions import (
    ACTION_NAMES,
    ACTION_SPACE_ID,
    LOWLEVEL_ACTION_MAP,
    N_LOWLEVEL_ACTIONS,
    lowlevel_action_name,
    lowlevel_action_tuple,
)


def test_lowlevel_12_action_contract():
    assert ACTION_SPACE_ID == "lowlevel_12"
    assert N_LOWLEVEL_ACTIONS == 12
    assert len(LOWLEVEL_ACTION_MAP) == 12
    assert len(ACTION_NAMES) == 12
    assert ACTION_NAMES == (
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

    left = GameAction.MOVE_LEFT.index
    right = GameAction.MOVE_RIGHT.index
    rot_left = GameAction.ROTATE_LEFT.index
    rot_right = GameAction.ROTATE_RIGHT.index
    hard_drop = GameAction.HARD_DROP.index
    hold = GameAction.HOLD.index

    assert LOWLEVEL_ACTION_MAP == (
        (left,),
        (right,),
        (rot_right,),
        (rot_left,),
        (hold,),
        (hard_drop,),
        (left, rot_right),
        (right, rot_right),
        (left, rot_left),
        (right, rot_left),
        (left, hard_drop),
        (right, hard_drop),
    )


def test_lowlevel_action_lookup_validates_range():
    assert lowlevel_action_tuple(10) == LOWLEVEL_ACTION_MAP[10]
    assert lowlevel_action_name(10) == "L+drop"

    with pytest.raises(ValueError, match="out of range"):
        lowlevel_action_tuple(12)
    with pytest.raises(ValueError, match="out of range"):
        lowlevel_action_name(-1)
