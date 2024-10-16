import numpy as np
import torch
import torch.nn as nn

from torchconverter import TracedModule


torch.manual_seed(0)

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(83, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
        )

    def forward(self, obs):
        hidden = self.policy_net.forward(obs)
        return hidden



class MergedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_extractor = ValueNet()
        self.action_net = nn.Linear(256, 5, bias=True)

    def forward(self, obs):
        hidden = self.mlp_extractor.forward(obs)
        acs = self.action_net.forward(hidden)
        return acs



state_dict = torch.load("policy.pt")

m = MergedNet()
m.load_state_dict(state_dict, strict=False)

m.eval()

m = TracedModule(m)

test_input = torch.ones((83, )).unsqueeze(0)

with torch.no_grad():
    output = m.forward(test_input)
    print("output", output)

m.convert()
print(output)
