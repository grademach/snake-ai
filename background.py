import pygame


class Background:
    def __init__(self, screen: pygame.SurfaceType, size: int, pixels_per_square, show_grid=False,
                 color=(0, 0, 0)) -> None:
        self.size = size
        self.show_grid = show_grid
        self.color = color
        self.screen = screen

        self.pps = pixels_per_square
        self.grid_color = (127, 127, 127)

    def draw(self):
        self.screen.fill(self.color)
        if self.show_grid:
            for x in range(self.size):
                for y in range(self.size):
                    rect = pygame.Rect(x * self.pps, y * self.pps, self.pps, self.pps)
                    pygame.draw.rect(self.screen, self.grid_color, rect, 1, 1)
