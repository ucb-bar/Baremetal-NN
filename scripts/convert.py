import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import json

import numpy as np
from dataloader import Dataloader
from model import EncoderRNN


dataloader = Dataloader()

n_hidden = 32
model = EncoderRNN(dataloader.n_letters, n_hidden, dataloader.n_categories)

print("shape of network:", dataloader.n_letters, n_hidden, dataloader.n_categories)

model.load("./logs/model.pth")

#print(dict(model.state_dict()))

model_dict = dict(model.state_dict())


with open("weights.h", "w") as f:
    f.write("#include <float.h>\n")
    f.write("#include <stddef.h>\n")
    f.write("\n")
    

    for key in model_dict:
        print(key)
        array = model_dict[key].numpy()
        layer_name = key.split(".")[0]

        if ".weight" in key:
            # store the transposed array in advance
            array = array.T
            f.write("const static size_t {layer_name}_WEIGHT_T_ROWS = {value};\n".format(layer_name=layer_name.upper(), value=array.shape[0]))
            f.write("const static size_t {layer_name}_WEIGHT_T_COLS = {value};\n".format(layer_name=layer_name.upper(), value=array.shape[1]))
            f.write("const static float {layer_name}_WEIGHT_DATA[] = {{".format(layer_name=layer_name.upper()))
        elif ".bias" in key:
            f.write("const static size_t {layer_name}_BIAS_ROWS = {value};\n".format(layer_name=layer_name.upper(), value=1))
            f.write("const static size_t {layer_name}_BIAS_COLS = {value};\n".format(layer_name=layer_name.upper(), value=array.shape[0]))
            f.write("const static float {layer_name}_BIAS_DATA[] = {{".format(layer_name=layer_name.upper()))
            
        flat_array = np.ndarray.flatten(array)
        for i in range(flat_array.shape[0]-1):
            f.write("{value}, ".format(value=flat_array[i]))
        f.write("{value}".format(value=flat_array[flat_array.shape[0]-1]))
            
        f.write("};\n")
        f.write("\n")

    f.write("\n\n")



for key in model_dict:
    model_dict[key] = model_dict[key].tolist()


json.dump(model_dict, open("./logs/model.json", "w"))
