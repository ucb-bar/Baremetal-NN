import numpy as np

from nn import *
from operators import *
from weights import *

def forward(input):
    # Input
    input_out = input
    # Linear
    i2h_out = nn_linear(input_out, i2h_weight_transposed, i2h_bias)
    # Linear
    h2o_out = nn_linear(i2h_out, h2o_weight_transposed, h2o_bias)
    # Log Softmax
    softmax_out = nn_logsoftmax(h2o_out)
    return softmax_out, i2h_out
