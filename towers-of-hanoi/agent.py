from towers_game import Towers
from move_model import Move_model
import torch

class Agent():
    def __init__(self, disk_amount):
        self.move_model = Move_model(disk_amount)
        self.point = 0
        
        
    def pick_move(self, state):
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

    def set_points(self, additional_points):
        self.points += additional_points