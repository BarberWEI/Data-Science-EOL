import torch
import torch.nn as nn
import torch.nn.functional as F

class HanoiModel(nn.Module):
    def __init__(self, disk_amount, hidden_size=64):
        super().__init__()

        input_size = disk_amount * 3
        output_size = 6  

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

