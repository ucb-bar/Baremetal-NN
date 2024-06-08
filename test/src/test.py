import torch


# seed random number generator
torch.manual_seed(0)

batch = 1
out_features = 4
in_features = 3

l = torch.nn.Linear(in_features, out_features)

print(l.state_dict()["weight"].numpy().flatten())
print(l.state_dict()["bias"].numpy().flatten())


input = torch.ones(batch, in_features)

output = l.forward(input)

print(output)

