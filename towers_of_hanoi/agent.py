from towers_game import Tower
from move_model import HanoiModel
import torch

class Agent():
    def __init__(self, disk_amount):
        self.move_model = HanoiModel(disk_amount)
        self.point = 0
        self.moves_made = 0
        self.way_of_end = 0
        self.disk_amount = disk_amount
    def is_dead(self):
        return self.point < -1000    
    def reset(self):
        self.point = 0
        self.moves_made = 0
        self.way_of_end = 0

        
    def pick_move(self, state):
        with torch.no_grad():
            model_output = self.move_model.forward(state)
        
        # this list just means from first one to second one, 
        # and it is supposed to match the ideas of the model output
        move_list = [
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 2],
            [2, 0],
            [2, 1],
        ]
        
        # this just finds the maximum index of ai output,
        # total of 6 possible ones
        idx = torch.argmax(model_output).item()
        
        return move_list[idx]


    def get_model_weights(self):
        return self.move_model.state_dict()
    
    
    def set_model_weights(self, weights):
        self.move_model.load_state_dict(weights)

    def add_points(self, additional_points):
        self.point += additional_points