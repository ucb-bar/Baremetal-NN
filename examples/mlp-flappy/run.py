import numpy as np
import torch
import torch.nn as nn

# import our converter module
from baremetal_nn import TracedModule


# set the seed for reproducibility
torch.manual_seed(0)

# example network
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


# in the example application, the network is split into two parts:
# the mlp_extractor and the action_net
class MergedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_extractor = ValueNet()
        self.action_net = nn.Linear(256, 5, bias=True)

    def forward(self, obs):
        hidden = self.mlp_extractor.forward(obs)
        acs = self.action_net.forward(hidden)
        return acs


# load the trained policy checkpoint
state_dict = torch.load("policy.pt")

# create the merged network
m = MergedNet()
# load the trained policy checkpoint
m.load_state_dict(state_dict, strict=False)

# set the network to evaluation mode
m.eval()

# trace the network
m = TracedModule(m)

# test the network
test_input = torch.ones((83, )).unsqueeze(0)
with torch.no_grad():
    output = m.forward(test_input)
    print("output:", output)

# convert the network to our baremetal c runtime
m.convert()

print("Done.")
