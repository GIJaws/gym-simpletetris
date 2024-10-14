import pygame
from gym_simpletetris.tetris.base_renderer import BaseRenderer


class HumanRenderer(BaseRenderer):
    def __init__(self, width, height, obs_type, block_size=20, fps=60, visible_height=None, **kwargs):
        super().__init__(width, height, obs_type, **kwargs)
        self.block_size = block_size
        self.fps = fps
        self.visible_height = visible_height or height

        # Initialize Pygame and other attributes
        pygame.init()
        pygame.display.init()
        window_height = self.visible_height * self.block_size
        window_width = self.width * self.block_size
        self.window = pygame.display.set_mode((window_width, window_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

    def render(self, game_state):
        self.window.fill((0, 0, 0))  # Clear screen with black background

        # Use shared utility functions
        board_blocks = self.format_blocks(game_state)
        piece_blocks = self.render_piece_data(game_state.current_piece)
        ghost_piece_blocks = self.render_piece_data(game_state.get_ghost_piece())

        # Render the board
        self._render_blocks(self.format_blocks(game_state))
        # Render the ghost piece
        self._render_blocks(ghost_piece_blocks, ghost=True)
        # Render the current piece
        self._render_blocks(piece_blocks)
        # Render UI elements
        self._render_ui(game_state)

        pygame.display.flip()
        self.clock.tick(self.fps)

        return None

    def _render_blocks(self, blocks, ghost=False):
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

    def _render_ui(self, game_state):
        x_offset = self.width * self.block_size + 10
        y_offset = 10
        gap = 30

        # Render available information from GameState
        for key, value in [
            ("Score", game_state.score),
            ("Level", game_state.level),
            ("Lines Cleared", game_state.total_lines_cleared),
            ("FPS", round(self.clock.get_fps(), 2)),
        ]:
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

    def _render_piece_preview(self, piece, pos, label):
        preview_size = self.block_size * 4
        block_size = self.block_size
        spacing = 10  # Vertical spacing between pieces

        # Draw label
        self._render_text(label, pos)
        pos = (pos[0], pos[1] + spacing)

        # Draw background
        rect = pygame.Rect(pos[0], pos[1], preview_size, preview_size)
        pygame.draw.rect(self.window, (50, 50, 50), rect)
        pygame.draw.rect(self.window, (255, 255, 255), rect, 1)  # Border

        # Center the piece in the preview
        min_x = min(x for x, y in piece.shape)
        max_x = max(x for x, y in piece.shape)
        min_y = min(y for x, y in piece.shape)
        max_y = max(y for x, y in piece.shape)
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
