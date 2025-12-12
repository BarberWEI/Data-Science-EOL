import pygame
import torch
from towers_game import Tower
from agent import Agent
import time

# ---------------------
# CONFIG
# ---------------------
WIDTH = 800
HEIGHT = 500
PEG_WIDTH = 10
DISK_HEIGHT = 20
COLORS = [
    (200, 50, 50),
    (50, 200, 50),
    (50, 50, 200),
    (200, 200, 50),
    (200, 50, 200),
    (50, 200, 200),
    (150, 150, 150),
    (255, 128, 0),
]

FPS = 2  # Slow enough to watch moves
# ---------------------


def draw_tower(screen, tower):
    screen.fill((255, 255, 255))

    peg_x = [WIDTH // 4, WIDTH // 2, 3 * WIDTH // 4]
    peg_y = HEIGHT - 50

    # Draw pegs
    for x in peg_x:
        pygame.draw.rect(screen, (0, 0, 0), (x - PEG_WIDTH // 2, 100, PEG_WIDTH, 300))

    # Draw disks
    for peg_index, peg in enumerate(tower):
        for level, disk in enumerate(reversed(peg)):
            width = disk * 20
            disk_color = COLORS[(disk - 1) % len(COLORS)]

            rect = pygame.Rect(
                peg_x[peg_index] - width // 2,
                peg_y - level * DISK_HEIGHT,
                width,
                DISK_HEIGHT - 2
            )

            pygame.draw.rect(screen, disk_color, rect)

    pygame.display.update()


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tower of Hanoi Visualization")

    clock = pygame.time.Clock()

    # Load trained agent
    agent = Agent(8)
    agent.move_model.load_state_dict(torch.load("towers_of_hanoi/models/best_agent_4.pth"))
    agent.move_model.eval()

    # Create game
    env = Tower(8)
    state, _, _, _ = env.get_initial_state()
    state = state.unsqueeze(0)

    running = True
    solved = False
    step_count = 0

    draw_tower(screen, env.tower)
    clock.tick(FPS)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if solved:
            continue

        # Agent picks move
        move = agent.pick_move(state)
        from_peg, to_peg = move[0], move[1]

        state, reward, done, info = env.step(from_peg, to_peg)
        state = state.unsqueeze(0)

        step_count += 1
        print(f"Step {step_count}: {from_peg} → {to_peg}, reward={reward}")

        # Draw updated tower
        draw_tower(screen, env.tower)
        clock.tick(FPS)

        if done:
            print("Solved!" if reward > 0 else "Episode ended.")
            solved = True

    pygame.quit()


if __name__ == "__main__":
    main()

