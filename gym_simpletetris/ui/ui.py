import pygame


class UIComponent:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)

    def update(self, game_state):
        pass

    def draw(self, surface):
        pass


class HeldPiece(UIComponent):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.held_piece = None

    def update(self, game_state):
        self.held_piece = game_state.get("held_piece")

    def draw(self, surface):
        # Draw background
        pygame.draw.rect(surface, (50, 50, 50), self.rect)
        pygame.draw.rect(surface, (255, 255, 255), self.rect, 2)  # White border

        # Draw text
        font = pygame.font.Font(None, 24)
        text = font.render("Held", True, (255, 255, 255))
        surface.blit(text, (self.rect.x + 5, self.rect.y + 5))

        if self.held_piece:
            # Draw the held piece
            block_size = min(self.rect.width, self.rect.height) // 5
            for x, y in self.held_piece:
                pygame.draw.rect(
                    surface,
                    (200, 200, 200),
                    (
                        self.rect.x + (x + 2) * block_size,
                        self.rect.y + (y + 2) * block_size,
                        block_size,
                        block_size,
                    ),
                )
        else:
            # Draw "No Piece" text
            no_piece_text = font.render("No Piece", True, (255, 255, 255))
            text_rect = no_piece_text.get_rect(center=self.rect.center)
            surface.blit(no_piece_text, text_rect)


class PiecePreview(UIComponent):
    def update(self, game_state):
        self.next_piece = game_state.get("next_piece")  # Use .get() method


class ScoreDisplay(UIComponent):
    def update(self, game_state):
        self.score = game_state.get("score", 0)  # Use .get() method with default value


class UIManager:
    def __init__(self):
        self.components = []

    def add_component(self, component):
        self.components.append(component)

    def remove_component(self, component):
        self.components.remove(component)

    def update(self, game_state):
        for component in self.components:
            component.update(game_state)

    def draw(self, surface):
        for component in self.components:
            component.draw(surface)
