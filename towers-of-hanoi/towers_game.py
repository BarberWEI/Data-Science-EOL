import numpy as np
import copy

class Towers():
    def __init__(self, disk_amount):
        # this makes it so that initial tower is something like 8, 7, 6, 5, 4, 3, 2, 1 with two blank rows
        self.initial_tower = [[i for i in range(disk_amount, 0, -1)], [], []] 
        self.tower = copy.deepcopy(self.initial_tower)
        self.moves_made = 0
        self.target_tower = [[], [], [i for i in range(disk_amount, 0, -1)]]
     
    def reset(self):
        self.tower = copy.deepcopy(self.initial_tower)
        self.moves_made = 0
        
    def is_legal(self, from_loc, to_loc):
        if len(self.tower[from_loc]) == 0:
            return False
        if len(self.tower[to_loc]) == 0:
            return True
        # can't put bigger on smaller
        return self.tower[from_loc][-1] < self.tower[to_loc][-1]


    def step(self, from_loc, to_loc):
        if self.is_legal(from_loc, to_loc):
            disk = self.tower[from_loc].pop()
            self.tower[to_loc].append(disk)
            self.moves_made += 1
            reward = 1 if self.complteted() else 0 # Simple reward
            
            return self.get_tower(), reward, self.completed(), ":D"
        else:
            # heavily penalize ilegal moves
            return self.get_tower(), -100, False, "Illegal move"
 
        
    def get_tower(self):
        return copy.deepcopy(self.tower)
    
    def completed(self):
        return self.tower == self.target_tower
    
    