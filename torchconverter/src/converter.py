import numpy as np
import torch
import torch.nn
import torch.fx
import jinja2


INDENT = "  "

class TorchConverter(torch.fx.Interpreter):
    @staticmethod
    def toNumpy(tensor: torch.Tensor):
        return tensor.cpu().detach().contiguous().numpy()
    
    @staticmethod
    def toBytes(ndarray: np.ndarray):
        return ndarray.astype(np.float32).flatten().tobytes()

    def __init__(self, model):
        graph = torch.fx.Tracer().trace(model)
        
        # Does some checks to make sure the Graph is well-formed.
        graph.lint()
        
        gm: torch.fx.GraphModule = torch.fx.GraphModule(model, graph)

        super().__init__(gm)

        self.graph: torch.fx.Graph = graph
        self.model: torch.nn.Module = model
        self.gm: torch.fx.GraphModule = gm

        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader("templates")
        )

        self.model_template = self.env.get_template("model.c.in")

        self.output_directory = "."

        self.model_struct = ""
        self.model_init = ""
        self.model_forward = ""
        self.weight_content = b""
        self.model_struct += INDENT + "Tensor input;\n"
        
        self.prev_layer_out_name = "input"
        
        self.model_init += INDENT + "NN_initTensor(&model->{prev_layer_out_name}, {dim}, (size_t[]){shape}, DTYPE_F32, NULL);\n".format(
            prev_layer_out_name=self.prev_layer_out_name,
            dim=2,
            shape=(1, 28*28)
        )
        
    def print(self):
        self.gm.graph.print_tabular()
        
    def getModuleInSequential(self, module, indicies):
        if len(indicies) == 0:
            return module
        return self.getModuleInSequential(module[indicies[0]], indicies[1:])

    def getModule(self, module_name):
        if "." in module_name:
            # if we have nn.Sequential layers
            target_hierarchy = module_name.split(".")
            sequential_name = target_hierarchy[0]

            # indicies = target_hierarchy[1:]
            indicies = [int(x) for x in target_hierarchy[1:]]

            module = getattr(self.model, sequential_name)
            return self.getModuleInSequential(module, indicies)
        
        return getattr(self.model, module_name)
    
    def addDataTensor(self, tensor_name, data):
        self.model_struct += INDENT + "Tensor {tensor_name};\n".format(
            tensor_name=tensor_name
        )
        self.model_init += INDENT + "NN_initTensor(&model->{tensor_name}, {dim}, (size_t[]){{{shape}}}, DTYPE_F32, array_pointer);\n".format(
            tensor_name=tensor_name,
            dim=len(data.shape),
            shape=", ".join(str(x) for x in data.shape)
        )
        self.model_init += INDENT + "array_pointer += {increment};\n".format(
            increment=np.prod(data.shape)
        )
        self.weight_content += TorchConverter.toBytes(data)

    def addOutputTensor(self, tensor_name, shape):
        self.model_struct += INDENT + "Tensor {tensor_name};\n".format(
            tensor_name=tensor_name
        )
        self.model_init += INDENT + "NN_initTensor(&model->{tensor_name}, {dim}, (size_t[]){{{shape}}}, DTYPE_F32, NULL);\n".format(
            tensor_name=tensor_name,
            dim=len(shape),
            shape=", ".join(str(x) for x in shape)
        )
        self.prev_layer_out_name = tensor_name
    
    def addReLU(self, tensor_name):
        self.model_forward += INDENT + "NN_ReLUInplace_F32(&model->{tensor_name});\n".format(
            tensor_name=tensor_name,
        )
    
    def addReLU6(self, tensor_name):
        self.model_forward += INDENT + "NN_ReLU6Inplace_F32(&model->{tensor_name});\n".format(
            tensor_name=tensor_name,
        )

    def call_function(self, target, args, kwargs):
        # print("call function:", target)

        if target == torch.nn.functional.relu:
            self.model_forward += INDENT + "// F.relu\n"
            self.addReLU(self.prev_layer_out_name)
            
        elif target == torch.nn.functional.relu6:
            self.model_forward += INDENT + "// F.relu6\n"
            self.addReLU6(self.prev_layer_out_name)
        
        # elif target == torch.nn.functional.max_pool2d:
        #     self.model_forward += INDENT + "// F.max_pool2d\n"
        #     print("max_pool2d")
        
        # elif target == torch.flatten:
        #     self.model_forward += INDENT + "// torch.flatten\n"
        
        else:
            print("[WARNING] Unsupported function call:", target)
            
        
        self.model_forward += "\n"

        return super().call_function(target, args, kwargs)

    def call_method(self, target, args, kwargs):
        # print("call method:", target)
        return super().call_method(target, args, kwargs)

    def call_module(self, target, args, kwargs):
        print("call module:", target)

        module = self.getModule(target)
        layer_name = target.replace(".", "_")
        
        self.model_init += "\n"
        self.model_init += INDENT + "// {module}: {layer_name}\n".format(
                module=type(module),
                layer_name=layer_name
            )
    
        if type(module) == torch.nn.Linear:
            self.addDataTensor(
                "{layer_name}_weight".format(layer_name=layer_name), 
                TorchConverter.toNumpy(module.state_dict().get("weight"))
                )
            
            if module.bias is not None:
                self.addDataTensor(
                    "{layer_name}_bias".format(layer_name=layer_name),
                    TorchConverter.toNumpy(module.state_dict().get("bias"))
                    )
            
            batch_size = int(args[0].shape[0])
            self.addOutputTensor(
                "{layer_name}_out".format(layer_name=layer_name), 
                (batch_size, module.out_features)
                )
            
            self.model_forward += INDENT + "NN_Linear_F32(&model->{layer_name}_out, &model->{prev_layer_out_name}, {weight}, {bias});\n".format(
                prev_layer_out_name=self.prev_layer_out_name,
                layer_name=layer_name,
                weight="&model->{layer_name}_weight".format(layer_name=layer_name),
                bias="&model->{layer_name}_bias".format(layer_name=layer_name) if module.bias is not None else "NULL"
            )

        elif type(module) == torch.nn.BatchNorm2d:
            if module.weight is not None:
                self.addDataTensor(
                    "{layer_name}_weight".format(layer_name=layer_name), 
                    TorchConverter.toNumpy(module.state_dict().get("weight"))
                    )
            if module.bias is not None:
                self.addDataTensor(
                    "{layer_name}_bias".format(layer_name=layer_name),
                    TorchConverter.toNumpy(module.state_dict().get("bias"))
                    )
            if module.running_mean is not None:
                self.addDataTensor(
                    "{layer_name}_running_mean".format(layer_name=layer_name),
                    TorchConverter.toNumpy(module.state_dict().get("running_mean"))
                    )
            if module.running_var is not None:
                self.addDataTensor(
                    "{layer_name}_running_var".format(layer_name=layer_name),
                    TorchConverter.toNumpy(module.state_dict().get("running_var"))
                    )
                
            batch_size = int(args[0].shape[0])
            self.addOutputTensor(
                "{layer_name}_out".format(layer_name=layer_name), 
                (batch_size, module.num_features, args[0].shape[2], args[0].shape[3])
                )
            
            self.model_forward += INDENT + "NN_BatchNorm2D_F32(&model->{layer_name}_out, &model->{prev_layer_out_name}, {weight}, {bias}, {running_mean}, {running_var}, {eps});\n".format(
                prev_layer_out_name=self.prev_layer_out_name,
                layer_name=layer_name,
                weight="&model->{layer_name}_weight".format(layer_name=layer_name) if module.weight is not None else "NULL",
                bias="&model->{layer_name}_bias".format(layer_name=layer_name) if module.bias is not None else "NULL",
                running_mean="&model->{layer_name}_running_mean".format(layer_name=layer_name) if module.running_mean is not None else "NULL",
                running_var="&model->{layer_name}_running_var".format(layer_name=layer_name) if module.running_var is not None else "NULL",
                eps=module.eps
            )

        elif type(module) == torch.nn.Conv2d:
            if module.weight is not None:
                self.addDataTensor(
                    "{layer_name}_weight".format(layer_name=layer_name), 
                    TorchConverter.toNumpy(module.state_dict().get("weight"))
                    )
            if module.bias is not None:
                self.addDataTensor(
                    "{layer_name}_bias".format(layer_name=layer_name),
                    TorchConverter.toNumpy(module.state_dict().get("bias"))
                    )
            
            batch_size = int(args[0].shape[0])
            self.addOutputTensor(
                "{layer_name}_out".format(layer_name=layer_name), 
                (batch_size, module.out_channels, module.kernel_size[0], module.kernel_size[1])
                )
        
            self.model_forward += INDENT + """NN_Conv2D_F32(
    &model->{layer_name}_out, &model->{prev_layer_out_name},
    {weight}, {bias}, (size_t[]){{{kernel_size}}}, (size_t[]){{{stride}}}, {groups});\n""".format(
                prev_layer_out_name=self.prev_layer_out_name,
                layer_name=layer_name,
                weight="&model->{layer_name}_weight".format(layer_name=layer_name) if module.weight is not None else "NULL",
                bias="&model->{layer_name}_bias".format(layer_name=layer_name) if module.bias is not None else "NULL",
                kernel_size=", ".join(str(x) for x in module.kernel_size),
                stride=", ".join(str(x) for x in module.stride),
                groups=module.groups
            )
        
        elif type(module) == torch.nn.ReLU:
            self.addReLU(self.prev_layer_out_name)
        
        elif type(module) == torch.nn.ReLU6:
            self.addReLU6(self.prev_layer_out_name)

        else:
            print("[WARNING] Unsupported module call:", target)


        return super().call_module(target, args, kwargs)

    def convert(self, example_input):
        output = self.run(example_input)

        print("finished tracing the model")
        
        with open("model.c", "w") as f:
            f.write(self.model_template.render(
                model_struct=self.model_struct, 
                model_init=self.model_init,
                model_forward=self.model_forward
            ))
        
        with open("weights.bin", "wb") as f:
            f.write(self.weight_content)
        


        return output




if __name__ == "__main__":
    import torch.nn as nn
    import torch.nn.functional as F


    class MobileNetSkipAdd(nn.Module):
        def __init__(self):

            convlayer = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU6(inplace=True)
                )

            super(MobileNetSkipAdd, self).__init__()
            self.conv0 = nn.Sequential(
                convlayer
            )

        def forward(self, x):
            x = self.conv0(x)
            return x
        


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # 1 input image channel, 6 output channels, 5x5 square convolution
            # kernel
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=False)  # 5*5 from image dimension
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, input):
            # Convolution layer C1: 1 input image channel, 6 output channels,
            # 5x5 square convolution, it uses RELU activation function, and
            # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
            c1 = F.relu(self.conv1(input))
            # Subsampling layer S2: 2x2 grid, purely functional,
            # this layer does not have any parameter, and outputs a (N, 6, 14, 14) Tensor
            s2 = F.max_pool2d(c1, (2, 2))
            # Convolution layer C3: 6 input channels, 16 output channels,
            # 5x5 square convolution, it uses RELU activation function, and
            # outputs a (N, 16, 10, 10) Tensor
            c3 = F.relu(self.conv2(s2))
            # Subsampling layer S4: 2x2 grid, purely functional,
            # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor
            s4 = F.max_pool2d(c3, 2)
            # Flatten operation: purely functional, outputs a (N, 400) Tensor
            s4 = torch.flatten(s4, 1)
            # Fully connected layer F5: (N, 400) Tensor input,
            # and outputs a (N, 120) Tensor, it uses RELU activation function
            f5 = F.relu(self.fc1(s4))
            # Fully connected layer F6: (N, 120) Tensor input,
            # and outputs a (N, 84) Tensor, it uses RELU activation function
            f6 = F.relu(self.fc2(f5))
            # Gaussian layer OUTPUT: (N, 84) Tensor input, and
            # outputs a (N, 10) Tensor
            output = self.fc3(f6)
            return output

    # Tracing the module
    m = MobileNetSkipAdd()

    m.forward(torch.ones(1, 3, 32, 32))

    TorchConverter(m).print()

    TorchConverter(m).convert(torch.ones(1, 3, 32, 32))
