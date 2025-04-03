import numpy as np
import torch
import torch.nn as nn

from baremetal_nn import TracedModule


torch.manual_seed(0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(48, 512, bias=True),
            nn.ELU(alpha=1.0),
            nn.Linear(512, 256, bias=True),
            nn.ELU(alpha=1.0),
            nn.Linear(256, 128, bias=True),
            nn.ELU(alpha=1.0),
            nn.Linear(128, 12, bias=True),
        )

    def forward(self, input):
        output = self.actor.forward(input)
        return output

# Tracing the module
m = Net()

# m.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
m.eval()

m = TracedModule(m)

test_input = torch.ones((48, )).unsqueeze(0)

with torch.no_grad():
    output = m.forward(test_input)
    print("output", output)

m.convert()
print(output)
