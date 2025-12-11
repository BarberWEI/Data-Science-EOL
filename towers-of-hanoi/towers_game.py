import numpy as np
import copy
import torch

class Tower():
    def __init__(self, disk_amount):
        # this makes it so that initial tower is something like 8, 7, 6, 5, 4, 3, 2, 1 with two blank rows
        self.disk_amount = disk_amount
        self.initial_tower = [[i for i in range(disk_amount, 0, -1)], [], []] 
        self.tower = copy.deepcopy(self.initial_tower)
        self.moves_made = 0
        self.min_moves = self.moves_to_solve()
        self.target_tower = [[], [], [i for i in range(disk_amount, 0, -1)]]
        self.previous_moves = self.moves_to_solve()
       
        
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

    def get_initial_state(self):
        return self.encode_state(), 0, self.completed(), ":D"
    
    def step(self, from_loc, to_loc):
        if self.is_legal(from_loc, to_loc):
            disk = self.tower[from_loc].pop()
            self.tower[to_loc].append(disk)
            self.moves_made += 1
            
            reward_multiplier = 0.1
            moves_left = self.moves_to_solve()
            # reward bot based on move performed as well as if they completed the puzzle
            
            reward = reward_multiplier * (self.previous_moves - moves_left)  
            self.previous_moves = moves_left
            reward += 100 if self.completed() else 0
            
            return self.encode_state(), reward, self.completed(), ":D"
        else:
            # heavily penalize ilegal moves
            return self.encode_state(), -10000, False, "Illegal move"
 
        
    def get_tower(self):
        return copy.deepcopy(self.tower)
    

    def encode_state(self):
        # this returns an encoded version of the towers game,
        # which makes it better for model training
        # essentially, each disk is one hot encoded so that 1,0,0 
        # means the disk is in the first peg
        endcoded = np.zeros(self.disk_amount * 3, dtype=np.float32)

        for peg_idx, peg in enumerate(self.tower):
            for disk in peg:
                endcoded[(disk - 1) * 3 + peg_idx] = 1.0
    
        return torch.tensor(endcoded)

    
    def completed(self):
        return self.tower == self.target_tower
    
    
    # GPT wrote this 
    def moves_to_solve(self, target=2):
        """
        state: [peg0_list, peg1_list, peg2_list]
        returns minimum number of moves to reach the solved configuration.
        """
        # Flatten with peg locations
        peg_of = {}
        for peg, lst in enumerate(self.tower):
            for disk in lst:
                peg_of[disk] = peg
        
        # Largest disk in puzzle
        n = max(peg_of.keys())

        # Recursive helper
        def solve_up_to(k, target):
            # If k == 0, nothing to move
            if k == 0:
                return 0
            
            # If disk k is already on target, skip it and check smaller disks
            if peg_of[k] == target:
                return solve_up_to(k - 1, target)
            
            # Identify auxiliary peg
            other_pegs = {0, 1, 2}
            src = peg_of[k]
            aux = list(other_pegs - {src, target})[0]

            # We must:
            # 1. Move disks 1..k-1 to aux peg
            # 2. Move disk k to target = 1 move
            # 3. Move disks 1..k-1 from aux to target (fully optimal = 2^(k-1)-1 moves)
            return (solve_up_to(k - 1, aux)
                    + 1
                    + (2**(k - 1) - 1))

        return solve_up_to(n, target)
