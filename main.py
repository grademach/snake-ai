import pygame
from background import Background
from snake import Snake
import helper

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
FPS = 15
GRID_SIZE = 25
PIXELS_PER_SQUARE = int(HEIGHT / GRID_SIZE)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class SnakeGame:
    def __init__(self, ai=False):
        self.render = True
        self.speed_up = False
        self.ai_controlled = ai
        self.snake = None
        # Sanity checks
        if WIDTH != HEIGHT:
            print("One or more constants failed sanity checks. Quitting process.")
            exit()

        # Create the game window
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake AI")

        # Create Game objects
        self.background = Background(self.screen, GRID_SIZE, PIXELS_PER_SQUARE, False)
        self.reset()

        # Clock to control the frame rate
        self.clock = pygame.time.Clock()

    def reset(self):
        self.snake = Snake(self.screen, PIXELS_PER_SQUARE, [int(GRID_SIZE / 2), int(GRID_SIZE / 2)], ai=self.ai_controlled, grid_size=GRID_SIZE)

    def tick(self, snake_action: list = None):
        if not self.snake.alive:
            self.reset()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                if not self.snake.ai and event.key in self.snake.move_keys:
                    self.snake.set_direction(helper.key_to_direction(event.key))
                if event.key == pygame.K_f:
                    self.speed_up = not self.speed_up
                if event.key == pygame.K_j:
                    self.render = not self.render

        # Game logic
        if snake_action is not None:
            self.snake.set_direction(action=snake_action)
        game_over, score, reward = self.snake.tick()

        # Render the game
        if self.render:
            self.background.draw()
            self.snake.draw()
            pygame.display.update()

        # Cap/Uncap the frame rate
        tick_rate = 0 if self.speed_up else FPS
        self.clock.tick(tick_rate)
        return game_over, score, reward


if __name__ == "__main__":
    game = SnakeGame()
    while True:
        game.tick()
