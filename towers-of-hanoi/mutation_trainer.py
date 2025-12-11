from towers_game import Tower
from agent import Agent
import torch

tower = Tower(8)
agent = Agent(8)
agent_is_done = 0
round_number = 0
agent_max_moves = tower.min_moves * 3
for _ in range(agent_max_moves):
    state = tower.get_initial_state()
    tower_state = state[0].unsqueeze(0)

    move = agent.pick_move(tower_state)
    state = tower.step(move[0],move[1])
    agent.add_points(state[1])
    tower_state = state[0].unsqueeze(0)

    if tower.completed():
        agent_is_done = 1
        break
    elif agent.is_dead():
        agent_is_done = -1
        break