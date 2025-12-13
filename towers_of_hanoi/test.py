from visualize_hanoi import HanoiVisualizer
from towers_game import Tower
import pygame
from agent import Agent
import torch

disk_amount = 4

agent = Agent(disk_amount)
visualizer = HanoiVisualizer(800, 600)
tower = Tower(disk_amount)


model_state = torch.load("towers_of_hanoi/models/best_agent_4.pth", map_location="cpu")
agent.move_model.load_state_dict(model_state)
agent.move_model.eval()

solved = False
state = tower.get_initial_state()
tower_state = state[0].unsqueeze(0)

while not solved:
    move = agent.pick_move(tower_state)
    state = tower.step(move[0],move[1])
    agent.add_points(state[1])
    tower_state = state[0].unsqueeze(0)
    print(tower.tower)
    if tower.completed():
        agent.way_of_end = 1
        completed = True
        break
    elif agent.is_dead():
        agent.way_of_end = -1
        break
# visualizer.render(tower.tower)

# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

# visualizer.close()