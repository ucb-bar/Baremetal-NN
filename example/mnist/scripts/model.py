import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(1 * 28 * 28, 16, device=self.device)
        self.fc2 = nn.Linear(16, 16, device=self.device)
        self.fc3 = nn.Linear(16, 10, device=self.device)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

