import numpy as np
import torch
import torch.nn as nn


class Simple(nn.Module):
    """
    Simple model with one linear layer
    """

    def __init__(self, dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor):
        y = self.fc1(x)
        return y

# Create model
model = Simple(dim=3)

# Save model
# torch.save(model, "model.pth")

# Load model
model = torch.load("model.pth")

# store model weights as binary file
model_structure = list(model.named_modules())

w1 = model.state_dict().get("fc1.weight").numpy().T
b1 = model.state_dict().get("fc1.bias").numpy()

print("w1:\n", w1)
print("b1:\n", b1)

w1 = w1.astype(np.float32).flatten()
b1 = b1.astype(np.float32).flatten()

with open("model.bin", "wb") as f:
    f.write(b1.tobytes())
    f.write(w1.tobytes())

# Test model
test_input = np.array([
    [1.0, 2.0, 3.0],
    [1.0, 2.0, 3.0],
    [1.0, 2.0, 3.0]
    ], dtype=np.float32)
test_tensor = torch.tensor(test_input, dtype=torch.float32)

output = model.forward(test_tensor)
print("model result:")
print(output)

print("raw result:")
print(test_input @ w1 + b1)
