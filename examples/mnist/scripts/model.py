import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_MLP(nn.Module):
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

class MNIST_CNN(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.avg_pool2d(x, 4)
        return x.view(-1, x.size(1))
