import numpy as np
import pytest

from gym_simpletetris.core.board import Board
from gym_simpletetris.env.start_states import (
    apply_prepared_board_grid,
    build_rgb_grid,
    easy_clears,
    easy_clears_preset,
    stacked_board_grid,
)


def _gap_columns_bottom_to_top(grid, n_rows):
    bottom_rows = grid[-n_rows:]
    return [int(np.where(row == 0)[0][0]) for row in reversed(bottom_rows)]


def _assert_one_gap_per_row(rows, width):
    assert rows.sum(axis=1).tolist() == [width - 1] * len(rows)
    assert [int((row == 0).sum()) for row in rows] == [1] * len(rows)


def test_apply_prepared_board_grid_syncs_grid_and_rgb():
    board = Board.create_board(width=4, height=6, buffer_height=2)
    grid = np.zeros((8, 4), dtype=np.uint8)
    grid[-1, [0, 2]] = 1
    grid[-2, 1] = 9

    updated = apply_prepared_board_grid(board, grid, filled_color=(11, 22, 33))

    assert updated.grid.dtype == np.uint8
    assert updated.rgb_grid.dtype == np.uint8
    assert updated.grid.tolist() == (grid > 0).astype(np.uint8).tolist()
    assert tuple(updated.rgb_grid[-1, 0]) == (11, 22, 33)
    assert tuple(updated.rgb_grid[-2, 1]) == (11, 22, 33)
    assert tuple(updated.rgb_grid[0, 0]) == (0, 0, 0)
    assert board.grid.sum() == 0


def test_build_rgb_grid_rejects_non_rgb_color():
    grid = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="RGB triples"):
        build_rgb_grid(grid, filled_color=(1, 2))


def test_stacked_board_grid_has_clear_spawn_area_and_no_full_rows():
    grid = stacked_board_grid(
        width=10,
        height=20,
        buffer_height=5,
        seed=7,
        stack_height=8,
        fill_density=1.0,
        n_holes=3,
    )

    assert grid.shape == (25, 10)
    assert grid.dtype == np.uint8
    assert grid[:17].sum() == 0
    assert grid[17:].sum() > 0
    assert not grid.all(axis=1).any()


def test_easy_single_preset_has_one_gap_per_bottom_row_with_varied_gaps():
    grid = easy_clears_preset(
        "easy_single",
        width=10,
        height=20,
        buffer_height=5,
        seed=2,
        n_rows=4,
    )
    bottom_rows = grid[-4:]
    gap_columns = [int(np.where(row == 0)[0][0]) for row in bottom_rows]

    assert grid.shape == (25, 10)
    _assert_one_gap_per_row(bottom_rows, width=10)
    assert len(set(gap_columns)) == 4
    assert all(left != right for left, right in zip(gap_columns, gap_columns[1:]))


def test_easy_single_strict_avoids_adjacent_gap_columns():
    grid = easy_clears_preset(
        "easy_single_strict",
        width=2,
        height=20,
        buffer_height=5,
        seed=0,
        n_rows=8,
    )
    gap_columns = _gap_columns_bottom_to_top(grid, 8)

    _assert_one_gap_per_row(grid[-8:], width=2)
    assert all(left != right for left, right in zip(gap_columns, gap_columns[1:]))


def test_easy_single_loose_uses_independent_random_gaps():
    grid = easy_clears_preset(
        "easy_single_loose",
        width=3,
        height=20,
        buffer_height=5,
        seed=0,
        n_rows=8,
    )
    same_seed = easy_clears_preset(
        "easy_single_loose",
        width=3,
        height=20,
        buffer_height=5,
        seed=0,
        n_rows=8,
    )
    gap_columns = _gap_columns_bottom_to_top(grid, 8)

    assert np.array_equal(grid, same_seed)
    _assert_one_gap_per_row(grid[-8:], width=3)
    assert gap_columns == [2, 1, 1, 0, 0, 0, 0, 0]
    assert any(left == right for left, right in zip(gap_columns, gap_columns[1:]))


def test_easy_tetris_alias_uses_super_easy_shared_gap_for_requested_rows():
    grid = easy_clears_preset(
        "easy_tetris",
        width=10,
        height=20,
        buffer_height=5,
        seed=3,
        n_rows=8,
    )
    bottom_rows = grid[-8:]
    gap_columns = [int(np.where(row == 0)[0][0]) for row in bottom_rows]

    _assert_one_gap_per_row(bottom_rows, width=10)
    assert len(set(gap_columns)) == 1
    assert grid[:-8].sum() == 0


def test_easy_tetris_loose_groups_rows_and_allows_repeated_group_gaps():
    grid = easy_clears_preset(
        "easy_tetris_loose",
        width=3,
        height=20,
        buffer_height=5,
        seed=0,
        n_rows=12,
    )
    gap_columns = _gap_columns_bottom_to_top(grid, 12)
    groups = [gap_columns[i : i + 4] for i in range(0, len(gap_columns), 4)]
    group_gaps = [group[0] for group in groups]

    _assert_one_gap_per_row(grid[-12:], width=3)
    assert all(len(set(group)) == 1 for group in groups)
    assert group_gaps == [2, 1, 1]
    assert any(left == right for left, right in zip(group_gaps, group_gaps[1:]))


def test_easy_tetris_strict_groups_rows_and_avoids_repeated_group_gaps():
    grid = easy_clears_preset(
        "easy_tetris_strict",
        width=3,
        height=20,
        buffer_height=5,
        seed=0,
        n_rows=12,
    )
    same_seed = easy_clears_preset(
        "easy_tetris_strict",
        width=3,
        height=20,
        buffer_height=5,
        seed=0,
        n_rows=12,
    )
    gap_columns = _gap_columns_bottom_to_top(grid, 12)
    groups = [gap_columns[i : i + 4] for i in range(0, len(gap_columns), 4)]
    group_gaps = [group[0] for group in groups]

    assert np.array_equal(grid, same_seed)
    _assert_one_gap_per_row(grid[-12:], width=3)
    assert all(len(set(group)) == 1 for group in groups)
    assert all(left != right for left, right in zip(group_gaps, group_gaps[1:]))


def test_easy_clears_clear_errors_for_unsupported_modes():
    with pytest.raises(
        NotImplementedError,
        match=r"clear_sizes=\[1\].*clear_sizes=\[4\]",
    ):
        easy_clears(clear_sizes=[2])
    with pytest.raises(ValueError, match="requires clear_sizes"):
        easy_clears(clear_sizes=[4], preset="single")
    with pytest.raises(ValueError, match="gap_policy='same_column'"):
        easy_clears(clear_sizes=[4], preset="easy_tetris_strict", gap_policy="varied")
    with pytest.raises(ValueError, match="width must be >= 2"):
        easy_clears_preset("easy_single_strict", width=1)
    with pytest.raises(ValueError, match="unsupported easy clears preset"):
        easy_clears_preset("bogus")
