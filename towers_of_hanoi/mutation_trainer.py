from towers_game import Tower
from agent import Agent
import torch
import copy
import random


agent_amount = 1000
disk_amount = 5

def get_best_agents(agents):
    # https://stackoverflow.com/questions/8966538/syntax-behind-sortedkey-lambda
    top_20 = sorted(agents, key=lambda agent: agent.point, reverse=True)[:20]
    return top_20


def make_new_models(best_agents, total_new=agent_amount, mutation_rate=0.1, mutation_strength=0.05):
    new_agents = []

    while len(new_agents) < total_new:
        
        parent = random.choice(best_agents)
        child = Agent(disk_amount)
        
        # this is
        child.move_model.load_state_dict(copy.deepcopy(parent.move_model.state_dict()))

        # Mutate the wweights
        with torch.no_grad():
            for param in child.move_model.parameters():
                # random mask for mutation  
                mask = (torch.rand_like(param) < mutation_rate).float()
                noise = torch.randn_like(param) * mutation_strength
                param += mask * noise

        new_agents.append(child)

    return new_agents

towers = []
agents = []

for _ in range(agent_amount):
    towers.append(Tower(disk_amount))
    agents.append(Agent(disk_amount))

agent_max_moves = towers[0].min_moves * 2
completed = False

while not completed:
    # state = towers[0].get_initial_state()
    # tower_state = state[0].unsqueeze(0)
    for agent_idx in range(len(agents)):
        state = towers[agent_idx].get_initial_state()
        tower_state = state[0].unsqueeze(0)
        current_agent = agents[agent_idx]
        current_tower = towers[agent_idx]
        for _ in range(agent_max_moves):
            move = current_agent.pick_move(tower_state)
            state = current_tower.step(move[0],move[1])
            current_agent.add_points(state[1])
            tower_state = state[0].unsqueeze(0)

            if current_tower.completed():
                agents[agent_idx].way_of_end = 1
                completed = True if current_tower.moves_made == towers[0].min_moves else False
                break
            elif current_agent.is_dead():
                current_agent.way_of_end = -1
                break
            
        current_agent.moves_made = current_tower.moves_made
    best_agent = max(agents, key=lambda agent: agent.point)
    print(best_agent.point, "step 1")
    print(best_agent.moves_made, "step 2")
    print(best_agent.way_of_end, "step 3")
    agents = make_new_models(get_best_agents(agents))
    towers = [Tower(disk_amount) for _ in range(agent_amount)]

best_agent = max(agents, key=lambda agent: agent.point)
torch.save(best_agent.move_model.state_dict(), "best_agent.pth")

