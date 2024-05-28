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
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

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
