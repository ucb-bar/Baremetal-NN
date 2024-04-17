import numpy as np
import torch
import torch.nn as nn

class Simple(nn.Module):
    def __init__(self, dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor):
        y = self.fc1(x)
        return y


model = Simple(dim=3)

# torch.save(model, "model.pth")

model = torch.load("model.pth")

test_a = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
input_tensor = torch.tensor(test_a, dtype=torch.float32)

output = model.forward(input_tensor)

print(output)


model_structure = list(model.named_modules())


w1 = model.state_dict().get("fc1.weight").numpy().T
b1 = model.state_dict().get("fc1.bias").numpy()

print("w1:\n", w1)
print("b1:\n", b1)
print("result:\n", test_a @ w1 + b1)

w1 = w1.astype(np.float32).flatten()
b1 = b1.astype(np.float32).flatten()


with open("model.bin", "wb") as f:
    f.write(b1.tobytes())
    f.write(w1.tobytes())
