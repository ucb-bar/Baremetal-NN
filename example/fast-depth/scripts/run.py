import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(2)
        
    def forward(self, x: torch.Tensor):
        y = self.conv1(x)
        y = self.batchnorm1(y)
        return y

# Create model
model = CNN()

# Save model
torch.save(model, "model.pth")

# store model weights as binary file
model_structure = list(model.named_modules())

test_input = torch.ones((1, 1, 4, 4), dtype=torch.float32)

model.eval()

with torch.no_grad():
    output = model.forward(test_input)


conv1_w = model.state_dict().get("conv1.weight").contiguous().numpy()
# conv1_b = model.state_dict().get("conv1.bias").contiguous().numpy()

batchnorm1_w = model.state_dict().get("batchnorm1.weight").contiguous().numpy()
batchnorm1_b = model.state_dict().get("batchnorm1.bias").contiguous().numpy()
batchnorm1_m = model.state_dict().get("batchnorm1.running_mean").contiguous().numpy()
batchnorm1_v = model.state_dict().get("batchnorm1.running_var").contiguous().numpy()
batchnorm1_num_batches = model.state_dict().get("batchnorm1.num_batches_tracked").contiguous().numpy()

print("batchnorm1_w:")
print(batchnorm1_w)
print("batchnorm1_b:")
print(batchnorm1_b)
print("batchnorm1_m:")
print(batchnorm1_m)
print("batchnorm1_v:")
print(batchnorm1_v)
print("batchnorm1_num_batches:")
print(batchnorm1_num_batches)

print("conv1_w:")
print(conv1_w.shape)

# print("conv1_b:")
# print(conv1_b.shape)


with open("model.bin", "wb") as f:
    f.write(conv1_w.astype(np.float32).flatten().tobytes())


# Test model

test_input = torch.ones((1, 1, 4, 4), dtype=torch.float32)

with torch.no_grad():
    output = model.forward(test_input)

print("input:")
print(test_input.shape)
print(test_input)

print("output:")
print(output.shape)
print(output)
