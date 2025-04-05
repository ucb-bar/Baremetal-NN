import numpy as np
import torch
import torch.nn as nn

# import our converter module
from baremetal_nn import TracedModule


# set the seed for reproducibility
torch.manual_seed(0)


# load the trained policy checkpoint
m = torch.load("model_4000.pt")

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
