import torch

batch = 5
out_features = 2
in_features = 3

x = torch.ones((batch, in_features))
w = torch.ones((out_features, in_features))
b = torch.zeros((out_features))

y = x @ w.T + b

print(y)
