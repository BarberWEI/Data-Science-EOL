from towers_game import Tower
from agent import Agent
import torch
import copy
import random

def get_best_agents(agents):
    # https://stackoverflow.com/questions/8966538/syntax-behind-sortedkey-lambda
    top_20 = sorted(agents, key=lambda agent: agent.point, reverse=True)[:20]
    return top_20


def make_new_models(best_agents, total_new=100, mutation_rate=0.1, mutation_strength=0.1):
    new_agents = []

    while len(new_agents) < total_new:
        
        parent = random.choice(best_agents)
        child = Agent(3)
        
        # this is just 
        child.move_model.load_state_dict(copy.deepcopy(parent.move_model.state_dict()))

        # Mutate weights
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

for _ in range(100):
    towers.append(Tower(3))
    agents.append(Agent(3))


agent_max_moves = towers[0].min_moves * 3
completed = False

for _ in range(100):
    # state = towers[0].get_initial_state()
    # tower_state = state[0].unsqueeze(0)
    for agent_idx in range(len(agents)):
        state = towers[agent_idx].get_initial_state()
        tower_state = state[0].unsqueeze(0)
        for _ in range(agent_max_moves):
            move = agents[agent_idx].pick_move(tower_state)
            state = towers[agent_idx].step(move[0],move[1])
            agents[agent_idx].add_points(state[1])
            tower_state = state[0].unsqueeze(0)

            if towers[agent_idx].completed():
                agents[agent_idx].way_of_end = 1
                completed = True
                break
            elif agents[agent_idx].is_dead():
                agents[agent_idx].way_of_end = -1
                break
            
        agents[agent_idx].moves_made =towers[agent_idx].moves_made
    best_agent = max(agents, key=lambda agent: agent.point)
    print(best_agent.point, "step 1")
    print(best_agent.moves_made, "step 2")
    print(best_agent.way_of_end, "step 3")
    agents = make_new_models(get_best_agents(agents))
    towers = [Tower(3) for _ in range(100)]


    


