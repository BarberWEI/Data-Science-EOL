import pygame
import random
import sys
from Bacteria import Bacterium

WORLD_SIZE = 500       # world is 100x100 cells
BACTERIA_SIZE = 5         
NUM_BACTERIA = 500
FPS = 30

# Colors
WHITE = (255, 255, 255)
PURPLE = (255, 0, 255)
GREEN = (0, 150, 0)
RED = (180, 0, 0)
BLACK = (0, 0, 0)

def create_world(size):
    # Start with scattered food (0–3 units)
    food_grid = [[random.randint(0, 3) for _ in range(size)] for _ in range(size)]
    # Waste starts empty
    waste_grid = [[0 for _ in range(size)] for _ in range(size)]
    return food_grid, waste_grid


def draw_world(screen, food_grid, waste_grid):
    for x in range(WORLD_SIZE):
        for y in range(WORLD_SIZE):

            px = x * BACTERIA_SIZE
            py = y * BACTERIA_SIZE

            if food_grid[x][y] > 0:
                pygame.draw.rect(screen, GREEN, (px, py, BACTERIA_SIZE, BACTERIA_SIZE))
            elif waste_grid[x][y] > 0:
                pygame.draw.rect(screen, RED, (px, py, BACTERIA_SIZE, BACTERIA_SIZE))


def draw_bacteria(screen, bacteria):
    for b in bacteria:
        px = b.loc_x * BACTERIA_SIZE
        py = b.loc_y * BACTERIA_SIZE
        if b.type == 1:
            pygame.draw.rect(screen, WHITE, (px, py, BACTERIA_SIZE, BACTERIA_SIZE))
        else:
            pygame.draw.rect(screen, PURPLE, (px, py, BACTERIA_SIZE, BACTERIA_SIZE))



def main():
    pygame.init()
    screen = pygame.display.set_mode((WORLD_SIZE * BACTERIA_SIZE, WORLD_SIZE * BACTERIA_SIZE))
    pygame.display.set_caption("Bacteria Simulation")
    clock = pygame.time.Clock()

    food_grid, waste_grid = create_world(WORLD_SIZE)

    bacteria = [
        Bacterium(random.randrange(WORLD_SIZE), random.randrange(WORLD_SIZE))
        for _ in range(NUM_BACTERIA)
    ]

    while True:
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # --- Update simulation ---
        for b in bacteria[:]:
            b.move(WORLD_SIZE, waste_grid, food_grid)
            b.eat(food_grid, waste_grid)
            b.produce_waste(waste_grid, food_grid)
            if b.energy > 50:
                bacteria.append(b.split())
            if b.is_dead():
                bacteria.remove(b)

        # --- Drawing ---
        screen.fill(BLACK)
        draw_world(screen, food_grid, waste_grid)
        draw_bacteria(screen, bacteria)

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()