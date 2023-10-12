import numpy as np

def nn_linear(input, weight_T, bias):
    return np.matmul(input, weight_T) + bias

def nn_logsoftmax(input):
    return np.log(np.exp(input) / np.sum(np.exp(input), axis=1))
