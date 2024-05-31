import pygame
from enum import Enum
import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.00000001)

class Direction(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    NONE = 5


def key_to_direction(key: pygame.key):
    if key in [pygame.K_UP, pygame.K_w]:
        return Direction.UP
    elif key in [pygame.K_DOWN, pygame.K_s]:
        return Direction.DOWN
    elif key in [pygame.K_LEFT, pygame.K_a]:
        return Direction.LEFT
    elif key in [pygame.K_RIGHT, pygame.K_d]:
        return Direction.RIGHT


def direction_to_vector(direction: Direction) -> list:
    match direction:
        case Direction.UP:
            return [0, -1]
        case Direction.DOWN:
            return[0, 1]
        case Direction.LEFT:
            return [-1, 0]
        case Direction.RIGHT:
            return [1, 0]
        case Direction.NONE:
            return [0, 0]


def offset_head(head_position: (int, int), offset: int):
    head_x, head_y = head_position
    head_right = [head_x + offset, head_y]
    head_left = [head_x - offset, head_y]
    head_up = [head_x, head_y - offset]
    head_down = [head_x, head_y + offset]
    return head_up, head_down, head_right, head_left


def get_near_space(position: (int, int), radius: int):
    pos_x = position[0] - radius
    pos_y = position[1] - radius
    diameter = (radius * 2) + 1
    space = []

    # positions = [pos_x + i for i in range(diameter)]

    for x in range(diameter):
        for y in range(diameter):
            coord = [pos_x + x, pos_y + y]
            if coord == list(position):
                continue
            space.append(coord)

    return space
