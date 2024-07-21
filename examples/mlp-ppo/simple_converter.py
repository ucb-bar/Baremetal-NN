# (0): Linear(in_features=795, out_features=512, bias=True)
# (1): ELU(alpha=1.0)
# (2): Linear(in_features=512, out_features=256, bias=True)
# (3): ELU(alpha=1.0)
# (4): Linear(in_features=256, out_features=128, bias=True)
# (5): ELU(alpha=1.0)
# (6): Linear(in_features=128, out_features=14, bias=True)

import torch
import numpy as np


def elu(x, alpha=1.0):
    for i in range(x.shape[-1]):
        if x[0, i] < 0:
            x[0, i] = alpha * (torch.exp(x[0, i]) - 1)
    return x


class HackyActorCritic(torch.nn.Module):
    def __init__(self):
        super(HackyActorCritic, self).__init__()
        self.fc1 = torch.nn.Linear(795, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 14)
    
    def forward(self, x):
        x = self.fc1.forward(x)
        x = elu(x, alpha=1.0)
        x = self.fc2.forward(x)
        x = elu(x, alpha=1.0)
        x = self.fc3.forward(x)
        x = elu(x, alpha=1.0)
        x = self.fc4.forward(x)
        return x


hack_policy = HackyActorCritic()

hack_policy.fc1.weight.data = torch.ones_like(hack_policy.fc1.weight.data)
hack_policy.fc1.bias.data = torch.ones_like(hack_policy.fc1.bias.data)
hack_policy.fc2.weight.data = torch.ones_like(hack_policy.fc2.weight.data)
hack_policy.fc2.bias.data = torch.ones_like(hack_policy.fc2.bias.data)
hack_policy.fc3.weight.data = torch.ones_like(hack_policy.fc3.weight.data)
hack_policy.fc3.bias.data = torch.ones_like(hack_policy.fc3.bias.data)
hack_policy.fc4.weight.data = torch.ones_like(hack_policy.fc4.weight.data)
hack_policy.fc4.bias.data = torch.ones_like(hack_policy.fc4.bias.data) * 2

hack_policy.fc4.bias.data[1] = 0

weights = b""

# hack_policy.fc1.weight.data.cpu().numpy().flatten().tobytes()
weights += hack_policy.fc1.weight.data.cpu().numpy().astype(np.float32).flatten().tobytes()
weights += hack_policy.fc1.bias.data.cpu().numpy().astype(np.float32).flatten().tobytes()
weights += hack_policy.fc2.weight.data.cpu().numpy().astype(np.float32).flatten().tobytes()
weights += hack_policy.fc2.bias.data.cpu().numpy().astype(np.float32).flatten().tobytes()
weights += hack_policy.fc3.weight.data.cpu().numpy().astype(np.float32).flatten().tobytes()
weights += hack_policy.fc3.bias.data.cpu().numpy().astype(np.float32).flatten().tobytes()
weights += hack_policy.fc4.weight.data.cpu().numpy().astype(np.float32).flatten().tobytes()
weights += hack_policy.fc4.bias.data.cpu().numpy().astype(np.float32).flatten().tobytes()

print("fc4 bias", hack_policy.fc4.bias.data.cpu().numpy().astype(np.float32))
print(hack_policy.fc4.bias.data.cpu().numpy().astype(np.float32).flatten().tobytes())

with open("hack_policy.bin", "wb") as f:
    f.write(weights)
    print("Wrote hack_policy.bin")

test_input = torch.ones(1, 795) * 0.01
test_output = hack_policy.forward(test_input)
print(test_output)
# breakpoint()