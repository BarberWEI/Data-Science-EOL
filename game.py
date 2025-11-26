import pygame
import random
import sys
from Bacteria import Bacterium
import numpy as np
WORLD_SIZE = 300       # world is 100x100 cells
#BACTERIA_SIZE = 5         
NUM_BACTERIA = 1000
FPS = 30

# Colors
WHITE = (255, 255, 255)
PURPLE = (255, 0, 255)
GREEN = (0, 150, 0)
RED = (180, 0, 0)
BLACK = (0, 0, 0)
world_surface = pygame.Surface((WORLD_SIZE, WORLD_SIZE))

def update_world_surface(world_surface, food_grid, waste_grid):
    # convert world to a 2d RGB numpy array (VERY FAST)
    arr = np.zeros((WORLD_SIZE, WORLD_SIZE, 3), dtype=np.uint8)

    arr[food_grid > 0] = (0,150,0)    # green
    arr[waste_grid > 0] = (180,0,0)  # red

    pygame.surfarray.blit_array(world_surface, arr)

def create_world(size):
    #initial_grid = np.random.randint(0, 3, (size, size)).astype(np.float32)
    food_grid = np.random.randint(0, 3, (size, size)).astype(np.float32)
    waste_grid = np.random.randint(0, 3, (size, size)).astype(np.float32)

    bacteria_grid = np.zeros((size, size), dtype=np.float32)
    bact_attack_grid = np.zeros((size, size), dtype=np.float32)
    phage_grid = np.zeros((size, size), dtype=np.float32)
    antibiotic_grid = np.zeros((size, size), dtype=np.float32)
    
    return food_grid, waste_grid, bacteria_grid, bact_attack_grid, phage_grid, antibiotic_grid


def draw_world(screen, food_grid, waste_grid, CELL_WIDTH, CELL_HEIGHT):
    for x in range(WORLD_SIZE):
        for y in range(WORLD_SIZE):
            px = x * CELL_WIDTH
            py = y * CELL_HEIGHT

            rect = (px, py, CELL_WIDTH, CELL_HEIGHT)

            if food_grid[x][y] > 0:
                pygame.draw.rect(screen, GREEN, rect)
            elif waste_grid[x][y] > 0:
                pygame.draw.rect(screen, RED, rect)


def draw_bacteria(screen, bacteria, CELL_WIDTH, CELL_HEIGHT):
    for b in bacteria:
        px = b.loc_x * CELL_WIDTH
        py = b.loc_y * CELL_HEIGHT

        rect = (px, py, CELL_WIDTH, CELL_HEIGHT)

        color = WHITE if b.type == 1 else PURPLE
        pygame.draw.rect(screen, color, rect)


def main():
    pygame.init()

    # FULLSCREEN MODE — ALWAYS FITS ANY MONITOR
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    screen_width, screen_height = screen.get_size()
    pygame.display.set_caption("Bacteria Simulation")
    clock = pygame.time.Clock()
    round_num = 0
    # Cell size — stretched to fill screen
    CELL_WIDTH = screen_width / WORLD_SIZE
    CELL_HEIGHT = screen_height / WORLD_SIZE

    food_grid, waste_grid, bacteria_grid, bact_attack_grid, phage_grid, antibiotic_grid = create_world(WORLD_SIZE)

    bacteria = [
        Bacterium(random.randrange(WORLD_SIZE), random.randrange(WORLD_SIZE))
        for _ in range(NUM_BACTERIA)
    ]

    while True:
        round_num += 1
        if len(bacteria) > 40000:
            break
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()

        # Reset bacteria grid
        bacteria_grid.fill(0)
        bact_attack_grid.fill(0)
 
        bxs = np.array([int(b.loc_x) for b in bacteria])
        bys = np.array([int(b.loc_y) for b in bacteria])
        types = np.array([b.type for b in bacteria])

        bacteria_grid.fill(0)
        bacteria_grid[bxs, bys] += np.where(types == 1, 1, -1)

        for b in bacteria[:]:
            b.emit_attack(bact_attack_grid, WORLD_SIZE)
            
        # Update bacteria
        should_make_waste = round_num % 5 == 0
        for b in bacteria[:]:
            move_decision = b.move(WORLD_SIZE, waste_grid, food_grid, bacteria_grid)
            b.eat(food_grid, waste_grid)
            
            b.produce_waste(waste_grid, food_grid) if should_make_waste else None
            
            b.check_immunity(round_num if round_num < 5000 else 5000)
            
            b.increase_age()
            
            damage = bact_attack_grid[b.loc_x][b.loc_y]

            b.energy -= damage
            b.energy += b.attack

            if b.is_dead():
                bacteria.remove(b)
                continue
            elif move_decision == 2:
                bacteria.append(b.split())
                
        oldest_bacteria = max(bacteria, key=lambda b: b.age) 
        most_splitty_bacteria = max(bacteria, key=lambda b: b.split_num)
        
        print(oldest_bacteria.age, most_splitty_bacteria.split_num)
        
        
        #print(round_num)
        # --- Drawing ---
        screen.fill(BLACK)
        update_world_surface(world_surface, food_grid, waste_grid)
        scaled = pygame.transform.scale(world_surface, screen.get_size())
        screen.blit(scaled, (0,0))
        
        draw_bacteria(screen, bacteria, CELL_WIDTH, CELL_HEIGHT)

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()