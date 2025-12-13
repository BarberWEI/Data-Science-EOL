from towers_game import Tower
from agent import Agent
import torch
import copy
import random


agent_amount = 10000
disk_amount = 4

def get_best_agents(agents):
    # https://stackoverflow.com/questions/8966538/syntax-behind-sortedkey-lambda
    top_20 = sorted(agents, key=lambda agent: agent.point, reverse=True)[:5]
    return top_20


def make_new_models(best_agents, total_new=agent_amount, mutation_rate=0.3, mutation_strength=0.1):
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
                # https://docs.pytorch.org/docs/stable/generated/torch.rand_like.html
                # https://medium.com/@jonas.schumacher/neural-network-layer-masking-in-pytorch-151dd834476e
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
runs = 0
while not completed and runs < 200:
    # state = towers[0].get_initial_state()
    # tower_state = state[0].unsqueeze(0)
    for agent_idx in range(len(agents)):
        state = towers[agent_idx].get_initial_state()
        tower_state = state[0].unsqueeze(0)
        current_agent = agents[agent_idx]
        current_tower = towers[agent_idx]
        #print("new agent")
        for _ in range(agent_max_moves):
            move = current_agent.pick_move(tower_state)
            state = current_tower.step(move[0],move[1])
            current_agent.add_points(state[1])
            tower_state = state[0].unsqueeze(0)
            #print(current_tower.tower)
            if state[2]:
                agents[agent_idx].way_of_end = 1
                completed = True 
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
    runs += 1
torch.save(best_agent.move_model.state_dict(), "towers_of_hanoi/models/best_agent.pth")

