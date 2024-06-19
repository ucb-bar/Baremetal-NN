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
    
    @staticmethod
    def dtypeToStr(dtype: torch.dtype):
        if dtype == torch.float16:
            return "DTYPE_F16"
        elif dtype == torch.float32:
            return "DTYPE_F32"

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
        self.placeholder_counter = {}
        self.function_counter = {}

        
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
    
    def addDataTensor(self, name, tensor):
        self.model_struct += INDENT + "Tensor {name};\n".format(
            name=name
        )
        data = TorchConverter.toNumpy(tensor)

        self.model_init += INDENT + "NN_initTensor(&model->{name}, {dim}, (size_t[]){{{shape}}}, {dtype}, array_pointer);\n".format(
            name=name,
            dim=len(tensor.shape),
            shape=", ".join(str(x) for x in tensor.shape),
            dtype=TorchConverter.dtypeToStr(tensor.dtype)
        )
        self.model_init += INDENT + "array_pointer += {increment};\n".format(
            increment=np.prod(tensor.shape)
        )
        self.weight_content += TorchConverter.toBytes(data)

    def addOutputTensor(self, name, shape, dtype=torch.float32):
        self.model_struct += INDENT + "Tensor {name};\n".format(
            name=name
        )
        self.model_init += INDENT + "NN_initTensor(&model->{name}, {dim}, (size_t[]){{{shape}}}, {dtype}, NULL);\n".format(
            name=name,
            dim=len(shape),
            shape=", ".join(str(x) for x in shape),
            dtype=TorchConverter.dtypeToStr(dtype)
        )
    
    def placeholder(self, target, args, kwargs):
        print("placeholder:", target)

        # this is also hacky
        
        # count = self.placeholder_counter.get(target, 0)
        # self.placeholder_counter[target] = count + 1

        name = target
        if name == "input":
            name = "input_1"
        
        self.model_struct += INDENT + "Tensor {name};\n".format(name=name)
        
        self.model_init += INDENT + "NN_initTensor(&model->{name}, {dim}, (size_t[]){{{shape}}}, DTYPE_F32, NULL);\n".format(
            name=name,
            dim=len(self.example_input.shape),
            shape=", ".join(str(x) for x in self.example_input.shape)
        )

        return super().placeholder(target, args, kwargs)
    
    def call_function(self, target, args, kwargs):
        # print("call function:", target)

        count = self.function_counter.get(target, 0)
        self.function_counter[target] = count + 1

        if target == operator.__add__:
            layer_name = "add_{count}".format(count=count) if count > 0 else "add"
            self.model_forward += INDENT + "// F.{layer_name}\n".format(layer_name=layer_name)
            self.model_forward += INDENT + "NN_add(&model->{layer_name}, &model->{input_names[0]}, &model->{input_names[1]});\n".format(
                layer_name=layer_name,
                input_names=self.node_info[layer_name][0]
            )
            self.addOutputTensor(layer_name, args[0].shape)
        
        elif target == torch.nn.functional.interpolate:
            layer_name = "interpolate_{count}".format(count=count) if count > 0 else "interpolate"
            self.model_forward += INDENT + "// F.{layer_name}\n".format(layer_name=layer_name)
            self.model_forward += INDENT + "NN_interpolate(&model->{layer_name}, &model->{input_names[0]}, (float []){{{scale_factor}, {scale_factor}}});\n".format(
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
            self.model_forward += INDENT + "NN_ReLU(&model->{layer_name}, &model->{input_names[0]});\n".format(
                layer_name=layer_name,
                input_names=self.node_info[layer_name][0]
            )
            
        elif target == torch.nn.functional.relu6:
            layer_name = "relu6_{count}".format(count=count) if count > 0 else "relu6"
            self.model_forward += INDENT + "// F.{layer_name}\n".format(layer_name=layer_name)
            self.model_forward += INDENT + "NN_ReLU6(&model->{layer_name}, &model->{input_names[0]});\n".format(
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
                module.state_dict().get("weight")
                )
            
            if module.bias is not None:
                self.addDataTensor(
                    "{layer_name}_bias".format(layer_name=layer_name),
                    module.state_dict().get("bias")
                    )
            
            batch_size = int(args[0].shape[0])
            self.addOutputTensor(
                layer_name,
                (batch_size, module.out_features)
                )
            
            self.model_forward += INDENT + "NN_Linear(&model->{layer_name}, &model->{input_names[0]}, {weight}, {bias});\n".format(
                layer_name=layer_name,
                input_names=input_names,
                weight="&model->{layer_name}_weight".format(layer_name=layer_name),
                bias="&model->{layer_name}_bias".format(layer_name=layer_name) if module.bias is not None else "NULL"
            )
        
        elif type(module) == torch.nn.BatchNorm2d:
            if module.weight is not None:
                self.addDataTensor(
                    "{layer_name}_weight".format(layer_name=layer_name), 
                    module.state_dict().get("weight")
                    )
            if module.bias is not None:
                self.addDataTensor(
                    "{layer_name}_bias".format(layer_name=layer_name),
                    module.state_dict().get("bias")
                    )
            if module.running_mean is not None:
                self.addDataTensor(
                    "{layer_name}_running_mean".format(layer_name=layer_name),
                    module.state_dict().get("running_mean")
                    )
            if module.running_var is not None:
                self.addDataTensor(
                    "{layer_name}_running_var".format(layer_name=layer_name),
                    module.state_dict().get("running_var")
                    )
                
            batch_size = int(args[0].shape[0])
            self.addOutputTensor(
                layer_name, 
                (batch_size, module.num_features, args[0].shape[2], args[0].shape[3])
                )
            
            self.model_forward += INDENT + """NN_BatchNorm2d(
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
                    module.state_dict().get("weight")
                    )
            if module.bias is not None:
                self.addDataTensor(
                    "{layer_name}_bias".format(layer_name=layer_name),
                    module.state_dict().get("bias")
                    )
            
            batch_size = int(args[0].shape[0])
            self.addOutputTensor(
                layer_name, 
                (batch_size, module.out_channels, args[0].shape[2]//module.stride[0], args[0].shape[3]//module.stride[1])
                )
        
            self.model_forward += INDENT + """NN_Conv2d(
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
            self.model_forward += INDENT + "NN_ReLU(&model->{layer_name}, &model->{input_names[0]});\n".format(
                layer_name=layer_name,
                input_names=input_names
            )
            self.addOutputTensor(layer_name, args[0].shape)
        
        elif type(module) == torch.nn.ReLU6:
            self.model_forward += INDENT + "NN_ReLU6(&model->{layer_name}, &model->{input_names[0]});\n".format(
                layer_name=layer_name,
                input_names=input_names
            )
            self.addOutputTensor(layer_name, args[0].shape)

        elif type(module) == torch.nn.ELU:
            self.model_forward += INDENT + "NN_ELU(&model->{layer_name}, &model->{input_names[0]}, {eps});\n".format(
                layer_name=layer_name,
                input_names=input_names,
                eps=module.alpha
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
            self.actor = nn.Sequential(
                nn.Linear(48, 5, bias=True),
                nn.ELU(),
                nn.Linear(5, 12, bias=True),
            )

        def forward(self, input):
            output = self.actor.forward(input)
            return output

    # Tracing the module
    m = Net()

    # m.load_state_dict(torch.load("mobilenet_skip_add.pth", map_location=torch.device('cpu')))
    m.eval()


    test_input = torch.zeros((48, )).unsqueeze(0)

    print(test_input)
    
    with torch.no_grad():
        output = m.forward(test_input)
        print(output)

    output = TorchConverter(m).convert(test_input)
    
