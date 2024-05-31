import pygame
import numpy as np
import helper
from helper import Direction, direction_to_vector
from random import randint


class Snake:
    def __init__(self, screen: pygame.SurfaceType, pixels_per_square: int, position=None, start_size=3,
                 color=(0, 255, 0), body_color=(0, 200, 90), ai=False, grid_size=None) -> None:
        if position is None:
            position = [0, 0]
        # AI Related
        self.ai = ai
        self.near_space_radius = 3
        self.near_space = helper.get_near_space(position, self.near_space_radius)

        # Base snake
        self.alive = True
        self.ticks_lived = 0
        self.position = position
        self.body = [[position[0] - (n + 1), position[1]] for n in range(start_size)]
        self.score = 0
        self.screen = screen
        self.color = color
        self.grid_size = grid_size
        self.body_color = body_color
        self.pps = pixels_per_square
        self.direction = Direction.RIGHT if self.ai else Direction.NONE
        self.last_direction = Direction.NONE
        self.apple_position = None
        self._spawn_apple()
        self.eat_apple_sound = pygame.mixer.Sound("./huap.wav")

        self.move_keys = [
            pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT,
            pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d
        ]
        self.directions_clockwise = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

    # Move and check for status changes
    def tick(self) -> (bool, int, int):
        """
        Tick the snake a single time
        :return: game_over, score, reward
        """
        reward = 0
        ate_apple = self._move(self.direction)
        self.near_space = helper.get_near_space(self.position, self.near_space_radius)
        if ate_apple:
            if not self.ai: pygame.mixer.Sound.play(self.eat_apple_sound)
            reward = 10

        should_die = self.check_danger(self.position)
        if should_die:
            self._die()  # self.alive = False
            reward = -10
        elif self.ai and self.ticks_lived > len(self.body) * 100:
            self._die()
            reward = -10

        # print(f"Tick, Pos: {self.position}, Body: {self.body}")
        if self.alive:
            self.ticks_lived += 1

        return not self.alive, self.score, reward

    def check_danger(self, position: [int, int], body_only=False):
        pos = not self.get_head_rect().colliderect(self.screen.get_rect())
        bod = position in self.body

        return bod if body_only else pos or bod

    def draw(self):
        # Draw snake body and head respectively
        for part in self.get_body_rects():
            pygame.draw.rect(self.screen, self.body_color, part, 0, int(self.pps / 4))
        pygame.draw.rect(self.screen, self.color, self.get_head_rect(), 0, int(self.pps / 4))

        # Draw apple
        apple = pygame.Rect(self.apple_position[0] * self.pps, self.apple_position[1] * self.pps, self.pps, self.pps)
        pygame.draw.rect(self.screen, (200, 0, 0), apple, 0, 15)

    def _move(self, direction: Direction) -> bool:
        """
        Move the snake in the current direction
        :return: Wether or not the snake ate an apple after moving
        """
        if not self.alive:
            return False
        vector = direction_to_vector(direction)
        if sum(vector) == 0:
            return False

        self.body.insert(0, self.position.copy())
        self.position[0] += vector[0]
        self.position[1] += vector[1]
        ate_apple = self.position == self.apple_position
        if ate_apple:
            self.score += 1
            if len(self.body) + 1 < np.power(self.grid_size, 2):
                self._spawn_apple()
        else:
            self.body.pop()
        self.last_direction = direction
        return ate_apple

    def set_direction(self, direction: Direction = None, action: list = None):
        if action is not None:
            index = self.directions_clockwise.index(self.direction)
            if np.array_equal(action, [1, 0, 0]):
                direction = self.directions_clockwise[index]
            elif np.array_equal(action, [0, 1, 0]):
                index = (index + 1) % len(self.directions_clockwise)
                direction = self.directions_clockwise[index]
            elif np.array_equal(action, [0, 0, 1]):
                direction = self.directions_clockwise[index-1]
            self.direction = direction

        elif direction is not None:
            match direction:
                case self.last_direction:
                    return
                case Direction.UP:
                    if self.last_direction == Direction.DOWN:
                        return
                case Direction.DOWN:
                    if self.last_direction == Direction.UP:
                        return
                case Direction.LEFT:
                    if self.last_direction in [Direction.NONE, Direction.RIGHT]:
                        return
                case Direction.RIGHT:
                    if self.last_direction == Direction.LEFT:
                        return
            self.direction = direction

    def get_head_rect(self) -> pygame.Rect:
        return self.rect_from_position(self.position)

    def get_body_rects(self) -> [pygame.Rect]:
        return [self.rect_from_position(pos) for pos in self.body]

    def rect_from_position(self, position: list) -> pygame.Rect:
        return pygame.Rect(position[0] * self.pps, position[1] * self.pps, self.pps, self.pps)

    def _die(self):
        if self.alive:
            self.alive = False
            self.color = (255, 0, 0)
            self.body_color = (130, 20, 20)

    def _spawn_apple(self):
        while True:
            pos = int(self.screen.get_height() / self.pps) - 1
            self.apple_position = [randint(0, pos), randint(0, pos)]
            snake_pos = self.body.copy()
            snake_pos.append(self.position.copy())
            if not (self.apple_position in snake_pos) and self.rect_from_position(self.apple_position).colliderect(self.screen.get_rect()):
                break
            # print("Failed to spawn apple, spawning again")

    def get_position_in_direction(self, direction: Direction):
        vector = direction_to_vector(direction)
        return [self.position[0] + vector[0], self.position[1] + vector[1]]

