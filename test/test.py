import torch


# seed random number generator
torch.manual_seed(0)

in_channels = 3
out_channels = 6
kernel_size = 5

x = torch.randn(1, in_channels, 64, 64)

conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
pool = torch.nn.MaxPool2d(2, 2)

y = conv.forward(x)
y_pool = pool.forward(y)

print(x.shape)
print(y.shape)
print(y_pool.shape)

