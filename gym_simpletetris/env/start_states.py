from __future__ import annotations

from dataclasses import replace
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from gym_simpletetris.core.board import Board
from gym_simpletetris.core.tetris_engine import GameState

DEFAULT_WIDTH = 10
DEFAULT_HEIGHT = 20
DEFAULT_BUFFER_HEIGHT = 5
DEFAULT_FILLED_RGB = (180, 180, 180)
DEFAULT_EMPTY_RGB = (0, 0, 0)

EasyClearsPreset = Literal["single", "tetris", "easy_single", "easy_tetris"]
GapPolicy = Literal["varied", "same_column"]
Grid = NDArray[np.uint8]


def build_rgb_grid(
    grid: NDArray[np.integer],
    *,
    filled_color: tuple[int, int, int] = DEFAULT_FILLED_RGB,
    empty_color: tuple[int, int, int] = DEFAULT_EMPTY_RGB,
) -> Grid:
    """Build renderer RGB data that matches a binary prepared board grid."""
    prepared = _as_grid(grid)
    filled = np.asarray(filled_color, dtype=np.uint8)
    empty = np.asarray(empty_color, dtype=np.uint8)
    if filled.shape != (3,) or empty.shape != (3,):
        raise ValueError("filled_color and empty_color must be RGB triples")

    rgb_grid = np.zeros((*prepared.shape, 3), dtype=np.uint8)
    rgb_grid[:, :] = empty
    rgb_grid[prepared > 0] = filled
    return rgb_grid


def apply_prepared_board_grid(
    board: Board,
    grid: NDArray[np.integer],
    *,
    filled_color: tuple[int, int, int] = DEFAULT_FILLED_RGB,
    empty_color: tuple[int, int, int] = DEFAULT_EMPTY_RGB,
) -> Board:
    """Return a board with prepared occupancy and matching RGB renderer data."""
    prepared = _as_grid(grid)
    expected_shape = (board.total_height, board.width)
    if prepared.shape != expected_shape:
        raise ValueError(f"prepared grid shape must be {expected_shape}, got {prepared.shape}")

    return replace(
        board,
        grid=prepared,
        rgb_grid=build_rgb_grid(prepared, filled_color=filled_color, empty_color=empty_color),
    )


def apply_prepared_board_grid_to_game_state(
    game_state: GameState,
    grid: NDArray[np.integer],
    *,
    filled_color: tuple[int, int, int] = DEFAULT_FILLED_RGB,
    empty_color: tuple[int, int, int] = DEFAULT_EMPTY_RGB,
) -> GameState:
    """Return a game state with prepared board occupancy and RGB kept in sync."""
    board = apply_prepared_board_grid(
        game_state.board,
        grid,
        filled_color=filled_color,
        empty_color=empty_color,
    )
    return replace(game_state, board=board)


def stacked_board_grid(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    buffer_height: int = DEFAULT_BUFFER_HEIGHT,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    stack_height: int = 14,
    fill_density: float = 0.85,
    n_holes: int = 4,
) -> Grid:
    """Build a dense bottom stack with holes and no immediately full rows."""
    _validate_dimensions(width, height, buffer_height)
    if stack_height < 1:
        raise ValueError("stack_height must be >= 1")
    if not 0.0 <= fill_density <= 1.0:
        raise ValueError("fill_density must be between 0.0 and 1.0")
    if n_holes < 0:
        raise ValueError("n_holes must be >= 0")

    generator = _coerce_rng(seed=seed, rng=rng)
    total_height = height + buffer_height
    stack_height = min(int(stack_height), height)
    stack_start = total_height - stack_height
    grid = np.zeros((total_height, width), dtype=np.uint8)

    for y in range(stack_start, total_height):
        row_fill = generator.random(width) < fill_density
        grid[y, row_fill] = 1
        if grid[y, :].all():
            grid[y, int(generator.integers(0, width))] = 0

    filled_positions = np.argwhere(grid == 1)
    if n_holes and len(filled_positions):
        hole_count = min(int(n_holes), len(filled_positions))
        for idx in generator.choice(len(filled_positions), size=hole_count, replace=False):
            y, x = filled_positions[int(idx)]
            grid[y, x] = 0

    return grid


def easy_clears(
    clear_sizes: Sequence[int],
    *,
    preset: EasyClearsPreset | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    buffer_height: int = DEFAULT_BUFFER_HEIGHT,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    n_rows: int = 4,
    gap_policy: GapPolicy | None = None,
) -> Grid:
    """Limited prepared-board generator for current easy single/tetris starts.

    Supported now:
    - clear_sizes=[1], preset single/easy_single: bottom rows each have one gap.
    - clear_sizes=[4], preset tetris/easy_tetris: bottom four rows share one gap.
    """
    _validate_dimensions(width, height, buffer_height)
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")

    normalized_sizes = tuple(int(size) for size in clear_sizes)
    kind = _resolve_easy_clears_kind(normalized_sizes, preset)
    policy = gap_policy or ("same_column" if kind == "tetris" else "varied")
    if kind == "single" and policy != "varied":
        raise ValueError("single easy_clears currently supports only gap_policy='varied'")
    if kind == "tetris" and policy != "same_column":
        raise ValueError("tetris easy_clears currently supports only gap_policy='same_column'")

    generator = _coerce_rng(seed=seed, rng=rng)
    total_height = height + buffer_height
    grid = np.zeros((total_height, width), dtype=np.uint8)

    if kind == "single":
        row_count = min(int(n_rows), height)
        gap_columns = _varied_gap_columns(width, row_count, generator)
    else:
        row_count = 4
        gap_column = int(generator.integers(0, width))
        gap_columns = [gap_column] * row_count

    for i, gap_column in enumerate(gap_columns):
        y = total_height - 1 - i
        grid[y, :] = 1
        grid[y, int(gap_column)] = 0

    return grid


def easy_clears_preset(
    kind: EasyClearsPreset,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    buffer_height: int = DEFAULT_BUFFER_HEIGHT,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    n_rows: int = 4,
) -> Grid:
    """Named facade for limited easy_clears presets."""
    if kind in ("single", "easy_single"):
        return easy_clears(
            clear_sizes=[1],
            preset="single",
            width=width,
            height=height,
            buffer_height=buffer_height,
            seed=seed,
            rng=rng,
            n_rows=n_rows,
        )
    if kind in ("tetris", "easy_tetris"):
        return easy_clears(
            clear_sizes=[4],
            preset="tetris",
            width=width,
            height=height,
            buffer_height=buffer_height,
            seed=seed,
            rng=rng,
        )
    raise ValueError(f"unsupported easy clears preset: {kind!r}")


def _as_grid(grid: NDArray[np.integer]) -> Grid:
    prepared = np.asarray(grid, dtype=np.uint8)
    if prepared.ndim != 2:
        raise ValueError(f"prepared grid must be 2D, got {prepared.ndim}D")
    return (prepared > 0).astype(np.uint8)


def _coerce_rng(*, seed: int | None, rng: np.random.Generator | None) -> np.random.Generator:
    if seed is not None and rng is not None:
        raise ValueError("pass seed or rng, not both")
    return rng if rng is not None else np.random.default_rng(seed)


def _resolve_easy_clears_kind(
    clear_sizes: tuple[int, ...],
    preset: EasyClearsPreset | None,
) -> Literal["single", "tetris"]:
    if preset in ("single", "easy_single"):
        if clear_sizes != (1,):
            raise ValueError("single easy_clears requires clear_sizes=[1]")
        return "single"
    if preset in ("tetris", "easy_tetris"):
        if clear_sizes != (4,):
            raise ValueError("tetris easy_clears requires clear_sizes=[4]")
        return "tetris"
    if preset is not None:
        raise ValueError(f"unsupported easy clears preset: {preset!r}")
    if clear_sizes == (1,):
        return "single"
    if clear_sizes == (4,):
        return "tetris"
    raise NotImplementedError(
        "easy_clears currently supports only clear_sizes=[1] and clear_sizes=[4]"
    )


def _varied_gap_columns(width: int, row_count: int, rng: np.random.Generator) -> list[int]:
    columns: list[int] = []
    while len(columns) < row_count:
        columns.extend(int(column) for column in rng.permutation(width))
    return columns[:row_count]


def _validate_dimensions(width: int, height: int, buffer_height: int) -> None:
    if width < 2:
        raise ValueError("width must be >= 2")
    if height < 1:
        raise ValueError("height must be >= 1")
    if buffer_height < 0:
        raise ValueError("buffer_height must be >= 0")
