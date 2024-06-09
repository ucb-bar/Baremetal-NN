import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)

DIM = 16


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor):
        y = nn.functional.relu(self.fc1(x))
        y = nn.functional.relu(self.fc2(x))
        y = self.fc3(x)
        return y

# Create model
model = MLP(dim=DIM)

# Save model
torch.save(model, "model.pth")

# Load model
# model = torch.load("model.pth")

# store model weights as binary file
model_structure = list(model.named_modules())

w1 = model.state_dict().get("fc1.weight").contiguous().numpy()
b1 = model.state_dict().get("fc1.bias").contiguous().numpy()
w2 = model.state_dict().get("fc2.weight").contiguous().numpy()
b2 = model.state_dict().get("fc2.bias").contiguous().numpy()
w3 = model.state_dict().get("fc3.weight").contiguous().numpy()
b3 = model.state_dict().get("fc3.bias").contiguous().numpy()

# print("w1:\n", w1)
# print("b1:\n", b1)

with open("model.bin", "wb") as f:
    f.write(w1.astype(np.float32).flatten().tobytes())
    f.write(b1.astype(np.float32).flatten().tobytes())
    f.write(w2.astype(np.float32).flatten().tobytes())
    f.write(b2.astype(np.float32).flatten().tobytes())
    f.write(w3.astype(np.float32).flatten().tobytes())
    f.write(b3.astype(np.float32).flatten().tobytes())



# Test model
test_input = np.ones((1, DIM), dtype=np.float32)

test_tensor = torch.tensor(test_input, dtype=torch.float32)

output = model.forward(test_tensor)
print("input:")
print(test_input)

print("output:")
# print(test_input @ w1.T + b1)
print(output)
