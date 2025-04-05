import numpy as np
import torch
import torch.nn as nn

from baremetal_nn import TracedModule


# set the seed for reproducibility
torch.manual_seed(0)

torch.set_printoptions(precision=3)


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


# prepare an example model
m = Net()

# set the model to evaluation mode
m.eval()

# wrap the model in our converter
m = TracedModule(m)

# we need to provide an example input to the model
# the value doesn't matter, as long as it has the right dimension and shape
test_input = torch.ones((48, )).unsqueeze(0)

# run the model with the example input
with torch.no_grad():
    output = m.forward(test_input)
    print("output", output)
    # tensor([[ 0.098,  0.041,  0.022, -0.045, -0.162, -0.034,  0.084,  0.035,  0.021,  0.102,  0.032,  0.047]])

# now the converter knows the dimension and shape of each layer
# we can convert the model to our baremetal C runtime
m.convert()

# the converted model is saved in `model.h`
# and the weights are saved in `model.bin`
print("Done.")
