import random
import torch
from Move_model import Move_model
from Points_model import Points_model

class Bacterium:
    def __init__(self, loc_x=0, loc_y=0, r=1):
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.r = r
        self.type = random.randint(0,1) # 1 means it eats food, 0 means it eats waste
        self.immunity_phage = 1
        self.immunity_antimicro = 1
        self.attack = 1
        self.energy = 10 if self.type == 1 else 10
        self.move_model = Move_model(77, [16, 16], 6)
        self.points = 0
        self.noise = torch.zeros((5,5))
        self.age = 0
        self.split_num = 0
        
        #self.noise = torch.randn((5, 5)) * 0.5
        with torch.no_grad():
            for param in self.move_model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        # outputs follow dx, dy, immune phage, attack, split, immune anti bact
        #self.points_model = Points_model(109, [64,16], 4)
        
    def get_local_tensor(self, food_grid, waste_grid, bacteria_grid, x, y, WORLD_SIZE):
        local_food = []
        local_waste = []
        local_bacteria = []

        offsets = [-2, -1, 0, 1, 2]

        for dy in offsets:
            food_row = []
            waste_row = []
            bacteria_row = []

            for dx in offsets:

                nx = (x + dx) % WORLD_SIZE
                ny = (y + dy) % WORLD_SIZE

                food_row.append(food_grid[nx][ny])
                waste_row.append(waste_grid[nx][ny])
                bacteria_row.append(bacteria_grid[nx][ny])


            local_food.append(food_row)
            local_waste.append(waste_row)
            local_bacteria.append(bacteria_row)
        

    
        # convert to PyTorch tensors
        food_tensor = torch.tensor(local_food, dtype=torch.float32) / 3.0 
        waste_tensor = torch.tensor(local_waste, dtype=torch.float32) / 3.0
        bacteria_tensor = torch.tensor(local_bacteria, dtype=torch.float32) / 5.0 
        energy_tensor = torch.tensor([self.energy / 30.0], dtype=torch.float32)
        type_tensor = torch.tensor([self.type], dtype=torch.float32)
        # loc_x = torch.tensor([self.loc_x / WORLD_SIZE], dtype=torch.float32)
        # loc_y = torch.tensor([self.loc_y / WORLD_SIZE], dtype=torch.float32)
        # stack into 3-channel tensor
        final_tensor = torch.cat([
            food_tensor.flatten(),
            waste_tensor.flatten(),
            bacteria_tensor.flatten(),
            energy_tensor,
            type_tensor,
            # loc_x,
            # loc_y
        ])


        return final_tensor
    
    
    def emit_attack(self, attack_grid, WORLD_SIZE):
        offsets = [-1, 0, 1]

        for dx in offsets:
            for dy in offsets:
                if dx == 0 and dy == 0:
                    continue  # do NOT hit yourself

                nx = int((self.loc_x + dx)) % WORLD_SIZE
                ny = int((self.loc_y + dy)) % WORLD_SIZE

                # Add your attack strength to the grid
                attack_grid[nx][ny] += self.attack


    def point_decisions(self, gain_immunity_phage, gain_attack, split, gain_immunity_antimicro):
        actions = [gain_immunity_phage, gain_attack, split, gain_immunity_antimicro]
        max_value = max(actions)
        max_index = actions.index(max_value)
        if max_value < 0.3:
            return

        #print(max_value, max_index)
        if max_index == 0 and self.points > 0:
            #print("did nothing")
            self.immunity_phage += 0.1
            self.points -= 1
            self.attack -= 0.005
            self.immunity_antimicro -= 0.005
            return 0
        elif max_index == 1 and self.points > 0:
            self.attack += 5
            self.points -= 1
            self.immunity_antimicro -= 0.005
            self.immunity_phage -= 0.005
            
            return 1
        elif max_index == 2:
            #print("split")
            self.points += 5
            return 2
        elif max_index == 3 and self.points > 0:
            self.immunity_antimicro += 0.1
            self.points -= 1
            self.attack -= 0.005
            self.immunity_phage -= 0.005
            return 3
        return 
        # elif max_index == 1 and self.points >= 1:
        #     return
        # elif max_index == 3  and self.points >= 1:
        #     return
        # elif max_index == 0 and self.points >= 1:
        #     return
    
    
    def check_immunity(self, round_num):
        threshold = 0.001 * round_num
        if self.immunity_antimicro < threshold or self.immunity_phage < threshold:
            self.energy -= 1
    
    
    def eat(self, food_grid, waste_grid):
        x, y = int(self.loc_x), int(self.loc_y)
        if self.type == 1 and food_grid[x][y] > 0:
            self.energy += 2
            food_grid[x][y] -= 1
            # waste_grid[x][y] += 1
        elif self.type == 0 and waste_grid[x][y] > 0:
            self.energy += 2
            waste_grid[x][y] -= 1
            # food_grid[x][y] += 1
        else:
            self.energy -= 3
            


    def produce_waste(self, waste_grid, food_grid):
        x, y = int(self.loc_x), int(self.loc_y)
        if self.type == 1:
            waste_grid[x][y] += 1 
        else:
            food_grid[x][y] += 1  
            
    def split(self):
        self.split_num += 1
        
        child = Bacterium(self.loc_x, self.loc_y)
        child.type = self.type
        child.immunity_phage = self.immunity_phage
        child.immunity_antimicro = self.immunity_antimicro 
        child.attack = self.attack / 2
        child.energy = self.energy / 3
        self.energy = self.energy / 3
        
        # copy parent's move_model weights with mutation
        mutation_std = 0.1
        child.move_model.load_state_dict(self.move_model.state_dict())

        # Then mutate child's weights
        with torch.no_grad():
            for param in child.move_model.parameters():
                param.add_(torch.randn_like(param) * mutation_std)

        return child

        
    def move(self, WORLD_SIZE, waste_grid, food_grid, bacteria_grid):
        local_tensor = self.get_local_tensor(food_grid, waste_grid, bacteria_grid, self.loc_x, self.loc_y, WORLD_SIZE)
        
        input_tensor = local_tensor.flatten().float().unsqueeze(0)
        actions = self.move_model(input_tensor)
        actions = actions.squeeze()
        #print(actions)
        dx = actions[0].item()
        dy = actions[1].item()
        
        dx = -1 if dx < -0.33 else (1 if dx > 0.33 else 0)
        dy = -1 if dy < -0.33 else (1 if dy > 0.33 else 0)
 
        self.loc_x = int(self.loc_x + dx) % WORLD_SIZE
        self.loc_y = int(self.loc_y + dy) % WORLD_SIZE



        return self.point_decisions(actions[2], actions[3], actions[4], actions[5])


    def is_dead(self):
        return self.energy <= 0 
    
    def increase_age(self):
        self.age += 1