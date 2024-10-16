import numpy as np
import torch
import torch.nn as nn

from torchconverter import TracedModule


torch.manual_seed(0)

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(83, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
        )

    def forward(self, obs):
        hidden = self.value.forward(obs)
        return hidden


class ActionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.action = nn.Linear(256, 5, bias=True)

    def forward(self, hidden):
        acs = self.action.forward(hidden)
        return acs


class MergedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = ValueNet()
        self.action = ActionNet()

    def forward(self, obs):
        hidden = self.value.forward(obs)
        acs = self.action.forward(hidden)
        return acs

# Tracing the module
m = MergedNet()

# m.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
m.eval()

m = TracedModule(m)

test_input = torch.ones((83, )).unsqueeze(0)

with torch.no_grad():
    output = m.forward(test_input)
    print("output", output)

m.convert()
print(output)
