
# adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import numpy as np
import torch

# from model import MLP

model = torch.load("model.pth")


for layer_name, module in model.named_modules():
    if type(module) == torch.nn.Linear:
        print("Linear Layer:", layer_name, type(module))
        print(module.in_features, module.out_features)
        print(module.weight, module.bias)


