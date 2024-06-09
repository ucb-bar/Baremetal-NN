import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm = nn.BatchNorm2d(2)
        
    def forward(self, x: torch.Tensor):
        y = self.conv1(x)
        y = self.batch_norm(y)
        return y

# Create model
model = CNN()

# Save model
torch.save(model, "model.pth")

# store model weights as binary file
model_structure = list(model.named_modules())

conv1_w = model.state_dict().get("conv1.weight").contiguous().numpy()
# conv1_b = model.state_dict().get("conv1.bias").contiguous().numpy()

print("conv1_w:")
print(conv1_w.shape)

# print("conv1_b:")
# print(conv1_b.shape)


with open("model.bin", "wb") as f:
    f.write(conv1_w.astype(np.float32).flatten().tobytes())


# Test model

test_input = torch.ones((1, 1, 4, 4), dtype=torch.float32)

output = model.forward(test_input)

print("input:")
print(test_input.shape)
print(test_input)

print("output:")
print(output.shape)
print(output)
