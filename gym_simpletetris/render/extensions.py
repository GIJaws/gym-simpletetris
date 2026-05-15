from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

RenderBlock = tuple[int, int, tuple[int, int, int]]


@dataclass(frozen=True)
class RenderContext:
    surface: Any
    game_state: Any
    board_rect: tuple[int, int, int, int]
    ui_rect: tuple[int, int, int, int]
    window_size: tuple[int, int]
    block_size: int
    board_width: int
    board_height: int
    visible_height: int
    fps: float


class RendererExtension(Protocol):
    def before_board(self, ctx: RenderContext) -> None: ...

    def render_ghost(self, ctx: RenderContext, blocks: Sequence[RenderBlock]) -> bool: ...

    def after_board(self, ctx: RenderContext) -> None: ...

    def before_ui(self, ctx: RenderContext) -> None: ...

    def after_ui(self, ctx: RenderContext) -> None: ...


class NoOpRendererExtension:
    def before_board(self, ctx: RenderContext) -> None:
        pass

    def render_ghost(self, ctx: RenderContext, blocks: Sequence[RenderBlock]) -> bool:
        return False

    def after_board(self, ctx: RenderContext) -> None:
        pass

    def before_ui(self, ctx: RenderContext) -> None:
        pass

    def after_ui(self, ctx: RenderContext) -> None:
        pass
