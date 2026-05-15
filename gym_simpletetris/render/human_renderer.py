import pygame
from gym_simpletetris.render.base_renderer import BaseRenderer
from gym_simpletetris.core.pieces import Piece
from gym_simpletetris.core.tetris_engine import GameState
from gym_simpletetris.render.extensions import RenderContext, RendererExtension


class HumanRenderer(BaseRenderer):
    def __init__(self, width, height, obs_type, block_size=20, fps=60, visible_height=None, extensions=None, **kwargs):
        super().__init__(width, height, obs_type, **kwargs)
        self.block_size = block_size
        self.fps = fps
        self.visible_height = visible_height or height
        self.extensions: list[RendererExtension] = list(extensions or [])

        # Initialize Pygame and other attributes
        pygame.init()
        pygame.display.init()
        window_height = 900
        window_width = 900
        self.window = pygame.display.set_mode((window_width, window_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

    def render(self, game_state: GameState):
        self.window.fill((0, 0, 0))  # ? Clear screen with black background

        ctx = self._render_context(game_state)
        for extension in tuple(self.extensions):
            extension.before_board(ctx)
        self._render_blocks(game_state.board.get_placed_blocks())
        ghost_blocks = game_state.get_ghost_piece().get_render_blocks()
        if not self._render_ghost_with_extensions(ctx, ghost_blocks):
            self._render_blocks(ghost_blocks, ghost=True)
        self._render_blocks(game_state.current_piece.get_render_blocks())
        for extension in tuple(self.extensions):
            extension.after_board(ctx)
        for extension in tuple(self.extensions):
            extension.before_ui(ctx)
        self._render_ui(game_state)
        for extension in tuple(self.extensions):
            extension.after_ui(ctx)

        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.fps)

        return None

    def add_extension(self, extension: RendererExtension) -> None:
        self.extensions.append(extension)

    def clear_extensions(self) -> None:
        self.extensions.clear()

    def _render_ghost_with_extensions(self, ctx: RenderContext, blocks) -> bool:
        for extension in tuple(self.extensions):
            render_ghost = getattr(extension, "render_ghost", None)
            if render_ghost is not None and render_ghost(ctx, blocks):
                return True
        return False

    def _render_context(self, game_state: GameState) -> RenderContext:
        window_width, window_height = self.window.get_size()
        board_width_px = self.width * self.block_size
        board_height_px = self.visible_height * self.block_size
        ui_x = board_width_px
        return RenderContext(
            surface=self.window,
            game_state=game_state,
            board_rect=(0, 0, board_width_px, board_height_px),
            ui_rect=(ui_x, 0, max(0, window_width - ui_x), window_height),
            window_size=(window_width, window_height),
            block_size=self.block_size,
            board_width=self.width,
            board_height=self.height,
            visible_height=self.visible_height,
            fps=float(self.clock.get_fps()),
        )

    def _render_blocks(self, blocks: list[tuple[int, int, tuple[int, int, int]]], ghost: bool = False) -> None:
        """
        Render a list of blocks on the game board.

        Args:
            blocks (list[tuple[int, int, tuple[int, int, int]]]): A list of tuples containing the x, y coordinates
                and the color of each block.
            ghost (bool, optional): Whether to render the blocks as a ghost piece (outline only). Defaults to False.
        """
        for x, y, color in blocks:
            # Adjust y to account for visible height
            y -= self.height - self.visible_height
            if y < 0 or y >= self.visible_height:
                continue  # Skip blocks outside the visible area

            rect = pygame.Rect(
                x * self.block_size,
                y * self.block_size,
                self.block_size,
                self.block_size,
            )
            if ghost:
                pygame.draw.rect(self.window, color, rect, 1)  # Outline for ghost piece
            else:
                pygame.draw.rect(self.window, color, rect)
                pygame.draw.rect(self.window, (50, 50, 50), rect, 1)  # Grid lines

    def _render_ui(self, game_state: GameState):
        x_offset = self.width * self.block_size + 10
        y_offset = 10
        gap = 25

        # Render available information from GameState
        info_to_display = [
            ("Score", game_state.score),
            ("Level", game_state.level),
            ("Lines Cleared", game_state.lines_cleared),
            ("Piece Timer", game_state.piece_timer),
            ("Gravity", f"Timer: {game_state.gravity_timer}, Interval: {round(game_state.gravity_interval, 2)}"),
            ("Lock Delay", f"{game_state.lock_delay_counter}/{game_state.MAX_LOCK_DELAY}"),
            ("Holes", game_state.board.count_holes()),
            ("Well Sums", sum(game_state.board.calculate_well_sums())),
            ("FPS", round(self.clock.get_fps(), 2)),
        ]
        for key, value in info_to_display:
            self._render_text(f"{key}: {value}", (x_offset, y_offset))
            y_offset += gap

        # Render held piece if available
        if game_state.held_piece:
            self._render_piece_preview(game_state.held_piece, (x_offset, y_offset), "Held Piece")
            y_offset += self.block_size * 5 + gap

        # Render next pieces
        for i, piece in enumerate(game_state.next_pieces[:5]):
            self._render_piece_preview(piece, (x_offset, y_offset), f"Next {i+1}")
            y_offset += self.block_size * 5 + gap

    def _render_text(self, text, pos):
        text_surface = self.font.render(text, True, (255, 255, 255))
        self.window.blit(text_surface, pos)

    def _render_piece_preview(self, piece: Piece, pos: tuple[int, int], label: str):
        preview_size = self.block_size * 4
        block_size = self.block_size
        spacing = 10  # Vertical spacing between pieces

        # Draw label
        self._render_text(label, pos)
        pos = (pos[0], pos[1] + 2 * spacing)

        # Draw background
        rect = pygame.Rect(pos[0], pos[1], preview_size, preview_size)
        pygame.draw.rect(self.window, (50, 50, 50), rect)
        pygame.draw.rect(self.window, (255, 255, 255), rect, 1)  # Border

        # Center the piece in the preview
        x_list = [x for x, _ in piece.shape]
        y_list = [y for _, y in piece.shape]
        min_x = min(x_list)
        max_x = max(x_list)
        min_y = min(y_list)
        max_y = max(y_list)
        width = max_x - min_x + 1
        height = max_y - min_y + 1

        offset_x = (preview_size - width * block_size) // 2
        offset_y = (preview_size - height * block_size) // 2

        for x, y in piece.shape:
            x_pos = pos[0] + offset_x + (x - min_x) * block_size
            y_pos = pos[1] + offset_y + (y - min_y) * block_size
            rect = pygame.Rect(x_pos, y_pos, block_size, block_size)
            pygame.draw.rect(self.window, piece.color, rect)
            pygame.draw.rect(self.window, (50, 50, 50), rect, 1)  # Grid lines

    def close(self):
        pygame.display.quit()
        pygame.quit()
