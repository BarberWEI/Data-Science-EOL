import pygame
import time

class HanoiVisualizer:
    def __init__(self, width=800, height=600, disk_colors=None):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Tower of Hanoi Visualizer")

        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)

        self.PEG_X = [width // 4, width // 2, 3 * width // 4]
        self.PEG_Y = height - 80

        # If no colors specified, auto-generate
        if disk_colors is None:
            self.disk_colors = {
                1: (255, 50, 50),
                2: (50, 255, 50),
                3: (50, 50, 255),
                4: (255, 200, 50),
                5: (255, 50, 200),
                6: (50, 255, 255),
                7: (180, 180, 180),
                8: (255, 128, 0)
            }
        else:
            self.disk_colors = disk_colors


    def draw_disk(self, peg_index, level, disk_size):
        x = self.PEG_X[peg_index]
        y = self.PEG_Y - level * 30
        radius = disk_size * 20

        color = self.disk_colors.get(disk_size, (200, 200, 200))
        pygame.draw.circle(self.screen, color, (x, y), radius)


    def draw_pegs(self):
        for x in self.PEG_X:
            pygame.draw.rect(self.screen, self.WHITE, (x - 5, 120, 10, 400))


    def render(self, tower, delay=0.5):
        """Draws the entire tower state."""
        self.screen.fill(self.BLACK)
        self.draw_pegs()

        for peg_index, peg in enumerate(tower):
            # Ensure drawing bottom→top
            disks = sorted(peg, reverse=True)
            for level, disk in enumerate(disks):
                self.draw_disk(peg_index, level, disk)

        pygame.display.flip()
        time.sleep(delay)


    def animate_move(self, from_peg, to_peg, tower, steps=15):
        """
        Animates a disk moving from one peg to another.
        tower gets modified outside; here we only animate.
        """
        disk = tower[from_peg][-1]  # disk being moved

        # Starting position
        start_x = self.PEG_X[from_peg]
        end_x = self.PEG_X[to_peg]

        # Vertical pickup height
        pickup_y = 80

        # Steps for smooth animation
        for i in range(steps):
            self.screen.fill(self.BLACK)
            self.draw_pegs()
            
            # Draw all other disks
            for peg_idx, peg in enumerate(tower):
                disks = sorted(peg, reverse=True)
                for level, d in enumerate(disks):
                    if d != disk:  # don't draw moving disk
                        self.draw_disk(peg_idx, level, d)

            # Animate horizontal movement
            t = i / steps
            x = start_x + (end_x - start_x) * t
            y = pickup_y

            pygame.draw.circle(self.screen, self.disk_colors[disk], (int(x), int(y)), disk * 20)

            pygame.display.flip()
            pygame.time.delay(30)


    def close(self):
        pygame.quit()
