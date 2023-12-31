import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import json
import shutil

from omegaconf import OmegaConf
import torch
import numpy as np

from dataloader import Dataloader
from model import EncoderRNN


conf = OmegaConf.load('./configs/eval.yaml')

class CodeConverter:
    class Target:
        Numpy = "numpy"
        C = "c"
    
    def __init__(self):
        self.model = None
    
    def load(self, model):
        print("loading model...")
        print(model)
        self.model = model
    
    def dump(self, out_path, target: Target = Target.C):
        weights_code, model_code = self.convert(target)
        
        if target == CodeConverter.Target.Numpy:
            file_extension = ".py"
        elif target == CodeConverter.Target.C:
            file_extension = ".h"
        
        # test if directory exists
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.mkdir(out_path)

        print("writing results to {path}...".format(path=out_path))
        with open(os.path.join(out_path, "nn"+file_extension), "w") as f:
            with open("libs/nn"+file_extension, "r") as nn_file:
                f.write(nn_file.read())
            
        with open(os.path.join(out_path, "weights"+file_extension), "w") as f:
            f.write(weights_code)

        with open(os.path.join(out_path, "model"+file_extension), "w") as f:
            f.write(model_code)
            

    def convert(self, target):
        if target == CodeConverter.Target.Numpy:
            print("dumping model as numpy...")
        elif target == CodeConverter.Target.C:
            print("dumping model as c...")
        
        # stores the content of network.xx
        model_code = ""
        # stores the content of weights.xx
        weights_code = ""

        if target == CodeConverter.Target.Numpy:
            INDENT = "    "
        elif target == CodeConverter.Target.C:
            INDENT = "  "

        if target == CodeConverter.Target.Numpy:
            weights_code += "import numpy as np\n"
            weights_code += "\n"

            model_code += "import numpy as np\n"
            model_code += "\n"
            model_code += "from nn import *\n"
            model_code += "from weights import *\n"
            model_code += "\n"
            model_code += "def forward(input):\n"
    
        elif target == CodeConverter.Target.C:
            weights_code += "#ifndef __WEIGHTS_H\n"
            weights_code += "#define __WEIGHTS_H\n"
            weights_code += "\n"
            weights_code += "#include <stdint.h>\n"
            weights_code += "#include <stddef.h>\n"
            weights_code += "#include <math.h>\n"
            weights_code += "#include <float.h>\n"
            weights_code += "\n"

            model_code += "#ifndef __MODEL_H\n"
            model_code += "#define __MODEL_H\n"
            model_code += "\n"
            model_code += "#include <stdint.h>\n"
            model_code += "#include <stddef.h>\n"
            model_code += "#include <math.h>\n"
            model_code += "#include <float.h>\n"
            model_code += "\n"
            model_code += "#include \"nn.h\"\n"
            model_code += "#include \"weights.h\"\n"
            model_code += "\n"
            model_code += "static Matrix i2h_weight_transposed = {\n"
            model_code += INDENT + ".rows = I2H_WEIGHT_TRANSPOSED_ROWS,\n"
            model_code += INDENT + ".cols = I2H_WEIGHT_TRANSPOSED_COLS,\n"
            model_code += INDENT + ".data = I2H_WEIGHT_TRANSPOSED_DATA\n"
            model_code += "};\n"
            model_code += "static Matrix i2h_bias = {\n"
            model_code += INDENT + ".rows = I2H_BIAS_ROWS,\n"
            model_code += INDENT + ".cols = I2H_BIAS_COLS,\n"
            model_code += INDENT + ".data = I2H_BIAS_DATA\n"
            model_code += "};\n"
            model_code += "static Matrix h2o_weight_transposed = {\n"
            model_code += INDENT + ".rows = H2O_WEIGHT_TRANSPOSED_ROWS,\n"
            model_code += INDENT + ".cols = H2O_WEIGHT_TRANSPOSED_COLS,\n"
            model_code += INDENT + ".data = H2O_WEIGHT_TRANSPOSED_DATA\n"
            model_code += "};\n"
            model_code += "static Matrix h2o_bias = {\n"
            model_code += INDENT + ".rows = H2O_BIAS_ROWS,\n"
            model_code += INDENT + ".cols = H2O_BIAS_COLS,\n"
            model_code += INDENT + ".data = H2O_BIAS_DATA\n"
            model_code += "};\n"
            model_code += "\n"
            model_code += "\n"
            model_code += "static void forward(Matrix *output, Matrix *input) {\n"


        prev_layer_name = "input"
        
        if target == CodeConverter.Target.Numpy:
            model_code += INDENT+"# Input\n"
            model_code += INDENT+"{layer_name}_out = input\n".format(layer_name=prev_layer_name)
        elif target == CodeConverter.Target.C:
            model_code += INDENT+"// Input\n"
            model_code += INDENT+"Matrix *{layer_name}_out = input;\n".format(layer_name=prev_layer_name)
        for layer_name, module in model.named_modules():
            print("Find network:", layer_name, type(module))

            if type(module) == type(model):
                # this is just wrapper of the PyTorch module, we should ignore
                continue
            if type(module) == torch.nn.Linear:
                for key in module.state_dict():
                    array = module.state_dict()[key].numpy()
                    
                    if key == "weight":
                        print("  ", key, "\t:", array.shape[0], "x", array.shape[1])
                        # store the transposed array in advance
                        array = array.T
                        
                        if target == CodeConverter.Target.Numpy:
                            weights_code += "{layer_name}_weight_rows = {value}\n".format(layer_name=layer_name, value=array.shape[0])
                            weights_code += "{layer_name}_weight_cols = {value}\n".format(layer_name=layer_name, value=array.shape[1])
                            weights_code += "{layer_name}_weight_transposed = np.array([".format(layer_name=layer_name)
                        elif target == CodeConverter.Target.C:
                            weights_code += "const static size_t {layer_name}_WEIGHT_TRANSPOSED_ROWS = {value};\n".format(layer_name=layer_name.upper(), value=array.shape[0])
                            weights_code += "const static size_t {layer_name}_WEIGHT_TRANSPOSED_COLS = {value};\n".format(layer_name=layer_name.upper(), value=array.shape[1])
                            weights_code += "const static float {layer_name}_WEIGHT_TRANSPOSED_DATA[] = {{".format(layer_name=layer_name.upper())
                    elif key == "bias":
                        print("  ", key, "\t:", array.shape[0])
                        if target == CodeConverter.Target.Numpy:
                            weights_code += "{layer_name}_bias_rows = {value}\n".format(layer_name=layer_name, value=1)
                            weights_code += "{layer_name}_bias_cols = {value}\n".format(layer_name=layer_name, value=array.shape[0])
                            weights_code += "{layer_name}_bias = np.array([".format(layer_name=layer_name)
                        elif target == CodeConverter.Target.C:
                            weights_code += "const static size_t {layer_name}_BIAS_ROWS = {value};\n".format(layer_name=layer_name.upper(), value=1)
                            weights_code += "const static size_t {layer_name}_BIAS_COLS = {value};\n".format(layer_name=layer_name.upper(), value=array.shape[0])
                            weights_code += "const static float {layer_name}_BIAS_DATA[] = {{".format(layer_name=layer_name.upper())
            
                    flat_array = np.ndarray.flatten(array)
                    for i in range(flat_array.shape[0]-1):
                        weights_code += "{value}, ".format(value=flat_array[i])
                    weights_code += "{value}".format(value=flat_array[flat_array.shape[0]-1])

                    if target == CodeConverter.Target.Numpy:
                        weights_code += "]).reshape({layer_name}_{key}_rows, {layer_name}_{key}_cols)\n".format(layer_name=layer_name, key=key)
                    elif target == CodeConverter.Target.C:
                        weights_code += "};\n"
                weights_code += "\n"

                if target == CodeConverter.Target.Numpy:
                    model_code += INDENT+"# Linear\n"
                    model_code += INDENT+"{layer_name}_out = nn_linear({prev_layer_name}_out, {layer_name}_weight_transposed, {layer_name}_bias)\n".format(layer_name=layer_name, prev_layer_name=prev_layer_name)
                elif target == CodeConverter.Target.C:
                    model_code += INDENT+"// Linear\n"
                    model_code += INDENT+"NN_linear({layer_name}_out, {layer_name}_weight_transposed, {layer_name}_bias, {prev_layer_name}_out);\n".format(layer_name=layer_name, prev_layer_name=prev_layer_name)
                
            if type(module) == torch.nn.LogSoftmax:
                if target == CodeConverter.Target.Numpy:
                    model_code += INDENT+"# Log Softmax\n"
                    model_code += INDENT+"{layer_name}_out = nn_logsoftmax({prev_layer_name}_out)\n".format(layer_name=layer_name, prev_layer_name=prev_layer_name)
                elif target == CodeConverter.Target.C:
                    model_code += INDENT+"// Log Softmax\n"
                    model_code += INDENT+"NN_logSoftmax({layer_name}_out, {prev_layer_name}_out);\n".format(layer_name=layer_name, prev_layer_name=prev_layer_name)

                
            prev_layer_name = layer_name
            
        if target == CodeConverter.Target.Numpy:
            model_code += INDENT+"return {prev_layer_name}_out\n".format(prev_layer_name=prev_layer_name)
        elif target == CodeConverter.Target.C:
            model_code += "}\n\n"
            model_code += "#endif  // __MODEL_H\n"
            weights_code += "\n\n"
            weights_code += "#endif  // __WEIGHTS_H\n"

        return weights_code, model_code



dataloader = Dataloader(conf.dataloader.path)
dataloader.load()

model = EncoderRNN(dataloader.n_letters, conf.model.hidden_layer_size, dataloader.n_categories)
model.load("./logs/model_h8_300000.pt")

print("shape of network:", dataloader.n_letters, conf.model.hidden_layer_size, dataloader.n_categories)


converter = CodeConverter()
converter.load(model)

path = "./runtime"

converter.dump(path, target=CodeConverter.Target.C)

