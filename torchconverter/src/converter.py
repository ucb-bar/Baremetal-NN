import operator

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

        # self.node_specs = [[n.op, n.name, n.target, n.args, n.kwargs] for n in self.graph.nodes]
        self.node_info = {n.name: (n.args, n.kwargs) for n in self.graph.nodes}

        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader("templates")
        )

        self.model_template = self.env.get_template("model.h.in")

        self.output_directory = "."

        self.model_struct = ""
        self.model_init = ""
        self.model_forward = ""
        self.weight_content = b""

        # this is sooooo hacky
        self.functional_counter = {}

        
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
    
    def placeholder(self, target, args, kwargs):
        print("placeholder:", target)

        # this is also hacky
        
        self.model_struct += INDENT + "Tensor {target};\n".format(target=target)
        
        self.model_init += INDENT + "NN_initTensor(&model->{target}, {dim}, (size_t[]){{{shape}}}, DTYPE_F32, NULL);\n".format(
            target=target,
            dim=len(self.example_input.shape),
            shape=", ".join(str(x) for x in self.example_input.shape)
        )

        return super().placeholder(target, args, kwargs)
    
    def call_function(self, target, args, kwargs):
        # print("call function:", target)

        count = self.functional_counter.get(target, 0)
        self.functional_counter[target] = count + 1

        if target == operator.__add__:
            layer_name = "add_{count}".format(count=count) if count > 0 else "add"
            self.model_forward += INDENT + "// F.{layer_name}\n".format(layer_name=layer_name)
            self.model_forward += INDENT + "NN_add_F32(&model->{layer_name}, &model->{input_names[0]}, &model->{input_names[1]});\n".format(
                layer_name=layer_name,
                input_names=self.node_info[layer_name][0]
            )
            self.addOutputTensor(layer_name, args[0].shape)
        
        elif target == torch.nn.functional.interpolate:
            layer_name = "interpolate_{count}".format(count=count) if count > 0 else "interpolate"
            self.model_forward += INDENT + "// F.{layer_name}\n".format(layer_name=layer_name)
            self.model_forward += INDENT + "NN_interpolate_F32(&model->{layer_name}, &model->{input_names[0]}, (float []){{{scale_factor}, {scale_factor}}});\n".format(
                layer_name=layer_name,
                input_names=self.node_info[layer_name][0],
                scale_factor=kwargs.get("scale_factor")
            )
            input_shape = args[0].shape
            self.addOutputTensor(
                layer_name, 
                (
                    input_shape[0], input_shape[1], 
                    input_shape[2]*kwargs.get("scale_factor"), input_shape[3]*kwargs.get("scale_factor")
                )
            )
        
        elif target == torch.nn.functional.relu:
            layer_name = "relu_{count}".format(count=count) if count > 0 else "relu"
            self.model_forward += INDENT + "// F.{layer_name}\n".format(layer_name=layer_name)
            self.model_forward += INDENT + "NN_ReLU_F32(&model->{layer_name}, &model->{input_names[0]});\n".format(
                layer_name=layer_name,
                input_names=self.node_info[layer_name][0]
            )
            
        elif target == torch.nn.functional.relu6:
            layer_name = "relu6_{count}".format(count=count) if count > 0 else "relu6"
            self.model_forward += INDENT + "// F.{layer_name}\n".format(layer_name=layer_name)
            self.model_forward += INDENT + "NN_ReLU6_F32(&model->{layer_name}, &model->{input_names[0]});\n".format(
                layer_name=layer_name,
                input_names=self.node_info[layer_name][0]
            )
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
        input_names = self.node_info[layer_name][0]
        
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
                layer_name,
                (batch_size, module.out_features)
                )
            
            self.model_forward += INDENT + "NN_Linear_F32(&model->{layer_name}, &model->{input_names}, {weight}, {bias});\n".format(
                layer_name=layer_name,
                input_names=input_names,
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
                layer_name, 
                (batch_size, module.num_features, args[0].shape[2], args[0].shape[3])
                )
            
            self.model_forward += INDENT + """NN_BatchNorm2d_F32(
    &model->{layer_name}, &model->{input_name[0]},
    {weight}, {bias}, 
    {eps}, {running_mean}, {running_var});\n""".format(
                layer_name=layer_name,
                input_name=input_names,
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
                layer_name, 
                (batch_size, module.out_channels, args[0].shape[2]//module.stride[0], args[0].shape[3]//module.stride[1])
                )
        
            self.model_forward += INDENT + """NN_Conv2d_F32(
    &model->{layer_name}, &model->{input_names[0]},
    {weight}, {bias}, (size_t[]){{{stride}}}, (size_t[]){{{padding}}}, {groups});\n""".format(
                layer_name=layer_name,
                input_names=input_names,
                weight="&model->{layer_name}_weight".format(layer_name=layer_name) if module.weight is not None else "NULL",
                bias="&model->{layer_name}_bias".format(layer_name=layer_name) if module.bias is not None else "NULL",
                stride=", ".join(str(x) for x in module.stride),
                padding=", ".join(str(x) for x in module.padding),
                groups=module.groups
            )
            self.prev_layer_name = "{layer_name}".format(layer_name=layer_name)
        
        elif type(module) == torch.nn.ReLU:
            self.model_forward += INDENT + "NN_ReLU_F32(&model->{layer_name}, &model->{input_names[0]});\n".format(
                layer_name=layer_name,
                input_names=input_names
            )
            self.addOutputTensor(layer_name, args[0].shape)
        
        elif type(module) == torch.nn.ReLU6:
            self.model_forward += INDENT + "NN_ReLU6_F32(&model->{layer_name}, &model->{input_names[0]});\n".format(
                layer_name=layer_name,
                input_names=input_names
            )
            self.addOutputTensor(layer_name, args[0].shape)

        else:
            print("[WARNING] Unsupported module call:", target)


        return super().call_module(target, args, kwargs)

    def convert(self, example_input):
        self.example_input = example_input
        output = self.run(example_input)

        print("finished tracing the model")
        
        with open("model.h", "w") as f:
            f.write(self.model_template.render(
                model_struct=self.model_struct, 
                model_init=self.model_init,
                model_forward=self.model_forward
            ))
        
        with open("model.bin", "wb") as f:
            f.write(self.weight_content)
        
        return output




if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)


    class MobileNetSkipAdd(nn.Module):
        def __init__(self):
            super(MobileNetSkipAdd, self).__init__()
            self.conv0 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
            )
            self.conv1 = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False),
                nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
                nn.Conv2d(16, 56, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(56, 56, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=56, bias=False),
                nn.BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
                nn.Conv2d(56, 88, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(88, 88, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=88, bias=False),
                nn.BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
                nn.Conv2d(88, 120, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(120, 120, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=120, bias=False),
                nn.BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
                nn.Conv2d(120, 144, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
            )
            self.conv5 = nn.Sequential(
                nn.Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False),
                nn.BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
                nn.Conv2d(144, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
            )
            self.conv6 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, bias=False),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
                nn.Conv2d(256, 408, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
            )
            self.conv7 = nn.Sequential(
                nn.Conv2d(408, 408, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=408, bias=False),
                nn.BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
                nn.Conv2d(408, 376, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(376, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
            )
            self.conv8 = nn.Sequential(
                nn.Conv2d(376, 376, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=376, bias=False),
                nn.BatchNorm2d(376, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
                nn.Conv2d(376, 272, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
            )
            self.conv9 = nn.Sequential(
                nn.Conv2d(272, 272, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=272, bias=False),
                nn.BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
                nn.Conv2d(272, 288, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
            )
            self.conv10 = nn.Sequential(
                nn.Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False),
                nn.BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
                nn.Conv2d(288, 296, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(296, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
            )
            self.conv11 = nn.Sequential(
                nn.Conv2d(296, 296, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=296, bias=False),
                nn.BatchNorm2d(296, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
                nn.Conv2d(296, 328, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(328, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
            )
            self.conv12 = nn.Sequential(
                nn.Conv2d(328, 328, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=328, bias=False),
                nn.BatchNorm2d(328, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
                nn.Conv2d(328, 480, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
            )
            self.conv13 = nn.Sequential(
                nn.Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False),
                nn.BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
                nn.Conv2d(480, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),
            )
            self.decode_conv1 = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 200, kernel_size=(1, 1), stride=(1, 1), bias=False),
                    nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                ),
            )
            self.decode_conv2 = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(200, 200, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=200, bias=False),
                    nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(200, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                ),
            )
            self.decode_conv3 = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=False),
                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 120, kernel_size=(1, 1), stride=(1, 1), bias=False),
                    nn.BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                ),
            )
            self.decode_conv4 = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False),
                    nn.BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(120, 56, kernel_size=(1, 1), stride=(1, 1), bias=False),
                    nn.BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                ),
            )
            self.decode_conv5 = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(56, 56, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=56, bias=False),
                    nn.BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(56, 16, kernel_size=(1, 1), stride=(1, 1), bias=False),
                    nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                ),
            )
            self.decode_conv6 = nn.Sequential(
                nn.Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            # skip connections: dec4: enc1
            # dec 3: enc2 or enc3
            # dec 2: enc4 or enc5
            for i in range(14):
                layer = getattr(self, 'conv{}'.format(i))
                x = layer(x)
                if i==1:
                    x1 = x
                elif i==3:
                    x2 = x
                elif i==5:
                    x3 = x
            for i in range(1,6):
                layer = getattr(self, 'decode_conv{}'.format(i))
                x = layer(x)
                x = F.interpolate(x, scale_factor=2, mode='nearest')
                if i==4:
                    x = x + x1
                elif i==3:
                    x = x + x2
                elif i==2:
                    x = x + x3
            x = self.decode_conv6(x)
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

    m.load_state_dict(torch.load("mobilenet_skip_add.pth", map_location=torch.device('cpu')))
    m.eval()



    # TorchConverter(m).print()

    # test_input = torch.zeros(1, 3, 224, 224)

    import cv2

    input_file = "../../example/cnn/visual_1.png"
    img = cv2.imread(input_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # convert to tensor
    img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0

    test_input = torch.tensor(img).unsqueeze(0)

    print(test_input)
    
    with torch.no_grad():
        output = m.forward(test_input)
        print(output)

    output = TorchConverter(m).convert(test_input)
    # print(output)
