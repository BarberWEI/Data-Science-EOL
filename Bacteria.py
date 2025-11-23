import random
import torch
from move_model import Move_model

class Bacterium:
    def __init__(self, loc_x=0, loc_y=0, r=1):
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.r = r
        self.type = random.randint(0,1) # 1 means it eats food, 0 means it eats waste
        self.energy = 10 if self.type == 1 else 20
        self.move_model = Move_model(72, 16, 2)
        
    def get_local_tensor(self, food_grid, waste_grid, x, y, WORLD_SIZE):
        local_food = []
        local_waste = []

        # we want a 6x6 grid centered around (x, y)
        # offsets will be -3, -2, -1, 0, 1, 2
        offsets = [-3, -2, -1, 0, 1, 2]

        for dy in offsets:
            food_row = []
            waste_row = []
            for dx in offsets:
                nx = (x + dx) % WORLD_SIZE   # wrap around world
                ny = (y + dy) % WORLD_SIZE

                food_row.append(food_grid[nx][ny])
                waste_row.append(waste_grid[nx][ny])

            local_food.append(food_row)
            local_waste.append(waste_row)

        # convert to PyTorch tensors (shape 6x6 each)
        food_tensor = torch.tensor(local_food, dtype=torch.float32)
        waste_tensor = torch.tensor(local_waste, dtype=torch.float32)

        # final shape: 2 channels × 6 × 6
        final_tensor = torch.stack([food_tensor, waste_tensor], dim=0)

        return final_tensor

    def eat(self, food_grid, waste_grid):
        if self.type == 1 and food_grid[self.loc_x][self.loc_y] > 0:
            self.energy += 3
            food_grid[self.loc_x][self.loc_y] -= 1
        elif self.type == 0 and waste_grid[self.loc_x][self.loc_y] > 0:
            self.energy += 2
            waste_grid[self.loc_x][self.loc_y] -= 1
        else:
            self.energy -= 1 if self.type == 1 else 0.1
            
            
    def split(self):
        child = Bacterium(self.loc_x, self.loc_y)
        child.type = self.type
        child.energy = self.energy / 2
        self.energy = self.energy / 2
        
        # copy parent's move_model weights with mutation
        mutation_std = 0.1
        child.move_model.load_state_dict(self.move_model.state_dict())

        # Then mutate child's weights
        with torch.no_grad():
            for param in child.move_model.parameters():
                param.add_(torch.randn_like(param) * mutation_std)

        return child
        
        
        
    def move(self, WORLD_SIZE, waste_grid, food_grid):
        local_tensor = self.get_local_tensor(food_grid, waste_grid, self.loc_x, self.loc_y, WORLD_SIZE)
        input_tensor = local_tensor.flatten().float().unsqueeze(0)
        dxdy = self.move_model(input_tensor)
        dxdy = dxdy.squeeze()
        print(dxdy)
        dx = dxdy[0].item()
        dy = dxdy[1].item()
        dx = int(round(dx))   # becomes -1, 0, +1
        dy = int(round(dy))
        self.loc_x = (self.loc_x + dx) % WORLD_SIZE
        self.loc_y = (self.loc_y + dy) % WORLD_SIZE
        
    def produce_waste(self, waste_grid, food_grid):
        if self.type == 1:
            waste_grid[self.loc_x][self.loc_y] += 1
        else:
            food_grid[self.loc_x][self.loc_y] += 1

    def is_dead(self):
        return self.energy <= 0
    
    