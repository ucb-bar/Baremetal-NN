import operator
import os
import inspect
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn
import torch.fx
import jinja2
import tabulate


INDENT = "  "


TEMPLATE_TENSOR_DECLARE = "Tensor {name};\n"
TEMPLATE_TENSOR_INIT = "NN_init_tensor(&model->{name}, {dim}, (size_t[]){{ {shape} }}, {dtype}, {data});\n"



def add_linear(ctx, layer_name, output_shape, input_names, weight, bias):
    ctx.add_data_tensor(
        "{layer_name}_weight".format(layer_name=layer_name), 
        weight
        )
    
    if bias is not None:
        ctx.add_data_tensor(
            "{layer_name}_bias".format(layer_name=layer_name),
            bias
            )
    
    ctx.add_output_tensor(
        layer_name,
        output_shape
        )
    
    ctx.model_forward += INDENT + "NN_linear(&model->{layer_name}, &model->{input_names[0]}, {weight}, {bias});\n".format(
        layer_name=layer_name,
        input_names=input_names,
        weight="&model->{layer_name}_weight".format(layer_name=layer_name),
        bias="&model->{layer_name}_bias".format(layer_name=layer_name) if bias is not None else "NULL"
    )






class TorchConverter(torch.fx.Interpreter):
    @staticmethod
    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a numpy array.
        
        Args:
            tensor (torch.Tensor): A PyTorch tensor.
        Returns:
            np.ndarray: A numpy array.
        """
        return tensor.cpu().detach().contiguous().numpy()
    
    @staticmethod
    def to_bytes(ndarray: np.ndarray) -> bytes:
        """
        Convert a numpy array to a bytes object.
        
        Args:
            ndarray (np.ndarray): A numpy array.
        Returns:
            bytes: A bytes object.
        """
        return ndarray.flatten().tobytes()
    
    @staticmethod
    def dtype_to_str(dtype: torch.dtype) -> str:
        """
        Convert a PyTorch dtype to a string.
        
        Args:
            dtype (torch.dtype): A PyTorch dtype.
        Returns:
            str: A string representation of the dtype.
        """
        if dtype == torch.uint8:
            return "DTYPE_U8"
        elif dtype == torch.int8:
            return "DTYPE_I8"
        elif dtype == torch.int16:
            return "DTYPE_I16"
        elif dtype == torch.int32:
            return "DTYPE_I32"
        elif dtype == torch.float16:
            return "DTYPE_F16"
        elif dtype == torch.float32:
            return "DTYPE_F32"
        return "UNKNOWN"

    @staticmethod
    def extract_graph_module(model: torch.nn.Module) -> list[torch.fx.Graph, torch.fx.GraphModule]:
        graph = torch.fx.Tracer().trace(model)
        # Does some checks to make sure the Graph is well-formed.
        graph.lint()
        gm = torch.fx.GraphModule(model, graph)
        return graph, gm

    @staticmethod
    def to_functional_torch(module: torch.nn.Module) -> Any:
        if type(module) == torch.nn.Linear:
            return torch.nn.functional.linear
        elif type(module) == torch.nn.Conv2d:
            return torch.nn.functional.conv2d
        elif type(module) == torch.nn.BatchNorm2d:
            return torch.nn.functional.batch_norm
        elif type(module) == torch.nn.ReLU:
            return torch.nn.functional.relu
        elif type(module) == torch.nn.ReLU6:
            return torch.nn.functional.relu6
        elif type(module) == torch.nn.ELU:
            return torch.nn.functional.elu
        else:
            print("[WARNING] Unsupported module call:", module)


    def __init__(self, model: torch.nn.Module):
        graph, gm = TorchConverter.extract_graph_module(model)
        super().__init__(gm)

        self.model: torch.nn.Module = model
        self.graph: torch.fx.Graph = graph
        self.gm: torch.fx.GraphModule = gm

        # extract node information
        self.node_info = {n.name: (n.args, n.kwargs) for n in self.graph.nodes}

        # initialize jinja2 code generation environment
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.path.join(
                os.path.dirname(inspect.getfile(inspect.currentframe())),
                "templates"))
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

    def print_graph(self):
        self.gm.graph.print_tabular()
        
    def get_module_in_sequential(self, module, indicies):
        if len(indicies) == 0:
            return module
        return self.get_module_in_sequential(module[indicies[0]], indicies[1:])

    def get_module(self, module_name):
        if "." in module_name:
            # if we have nn.Sequential layers
            target_hierarchy = module_name.split(".")
            sequential_name = target_hierarchy[0]

            # indicies = target_hierarchy[1:]
            indicies = [int(x) for x in target_hierarchy[1:]]

            module = getattr(self.model, sequential_name)
            return self.get_module_in_sequential(module, indicies)
        
        return getattr(self.model, module_name)
    
    def add_data_tensor(self, name, tensor):
        self.model_struct += INDENT + TEMPLATE_TENSOR_DECLARE.format(
            name=name
        )
        data = TorchConverter.to_numpy(tensor)

        self.model_init += INDENT + TEMPLATE_TENSOR_INIT.format(
            name=name,
            dim=len(tensor.shape),
            shape=", ".join(str(x) for x in tensor.shape),
            dtype=TorchConverter.dtype_to_str(tensor.dtype),
            data="weight_ptr"
        )
        self.model_init += INDENT + "weight_ptr += {increment};\n".format(
            increment=np.prod(tensor.shape)
        )
        self.weight_content += TorchConverter.to_bytes(data)

    def add_output_tensor(self, name, shape, dtype=torch.float32):
        self.model_struct += INDENT + TEMPLATE_TENSOR_DECLARE.format(
            name=name
        )
        self.model_init += INDENT + TEMPLATE_TENSOR_INIT.format(
            name=name,
            dim=len(shape),
            shape=", ".join(str(x) for x in shape),
            dtype=TorchConverter.dtype_to_str(dtype),
            data="NULL"
        )
    

    
    def trace_functional(self, layer_name, target, args, kwargs, out):
        print("  trace:", layer_name, target)
        
        output_shape = out.shape
        if len(output_shape) == 4:
            output_shape = (output_shape[0], output_shape[2], output_shape[3], output_shape[1])
        
        input_names = self.node_info[layer_name][0]

        if target == operator.__add__:
            self.model_forward += INDENT + "NN_add(&model->{layer_name}, &model->{input_names[0]}, &model->{input_names[1]});\n".format(
                layer_name=layer_name,
                input_names=input_names
            )
            self.add_output_tensor(layer_name, output_shape)
        elif target == torch.nn.functional.interpolate:
            self.model_forward += INDENT + "NN_interpolate(&model->{layer_name}, &model->{input_names[0]}, (float []){{{scale_factor}, {scale_factor}}});\n".format(
                layer_name=layer_name,
                input_names=input_names,
                scale_factor=kwargs.get("scale_factor")
            )
            self.add_output_tensor(layer_name, output_shape)
        elif target == torch.nn.functional.relu:
            self.model_forward += INDENT + "NN_relu(&model->{layer_name}, &model->{input_names[0]});\n".format(
                layer_name=layer_name,
                input_names=input_names
            )
            self.add_output_tensor(layer_name, output_shape)
        elif target == torch.nn.functional.relu6:
            self.model_forward += INDENT + "NN_relu6(&model->{layer_name}, &model->{input_names[0]});\n".format(
                layer_name=layer_name,
                input_names=input_names
            )
            self.add_output_tensor(layer_name, output_shape)
        elif target == torch.nn.functional.conv2d:
            weight = args[1]
            bias = args[2]
            stride = args[3]
            padding = args[4]
            dilation = args[5]
            groups = args[6]
            
            if weight is not None:
                # weight need to be converted from (out_ch, in_ch, kh, kw) to (kh, kw, in_ch, out_ch)
                self.add_data_tensor(
                    "{layer_name}_weight".format(layer_name=layer_name), 
                    weight.permute(2, 3, 1, 0)
                    )
            if bias is not None:
                self.add_data_tensor(
                    "{layer_name}_bias".format(layer_name=layer_name),
                    bias
                    )
            
            self.add_output_tensor(layer_name, output_shape)
        
            self.model_forward += INDENT + """NN_conv2d(
    &model->{layer_name}, &model->{input_names[0]},
    {weight}, {bias}, (size_t[]){{{stride}}}, (size_t[]){{{padding}}}, (size_t[]){{{dilation}}}, {groups});\n""".format(
                layer_name=layer_name,
                input_names=input_names,
                weight="&model->{layer_name}_weight".format(layer_name=layer_name) if weight is not None else "NULL",
                bias="&model->{layer_name}_bias".format(layer_name=layer_name) if bias is not None else "NULL",
                stride=", ".join(str(x) for x in stride),
                padding=", ".join(str(x) for x in padding),
                dilation=", ".join(str(x) for x in dilation),
                groups=groups
            )
            self.prev_layer_name = "{layer_name}".format(layer_name=layer_name)
        
        elif target == torch.nn.functional.linear:
            input_names = input_names
            weight = args[1]
            bias = args[2]
            self.model_forward += INDENT + "NN_linear(&model->{layer_name}, &model->{input_names[0]}, {weight}, {bias});\n".format(
                layer_name=layer_name,
                input_names=self.node_info[layer_name][0],
                weight="&model->{layer_name}_weight".format(layer_name=layer_name),
                bias="&model->{layer_name}_bias".format(layer_name=layer_name)
            )
            if weight is not None:
                # weight need to be converted from (out_ch, in_ch, kh, kw) to (kh, kw, in_ch, out_ch)
                self.add_data_tensor(
                    "{layer_name}_weight".format(layer_name=layer_name), 
                    weight
                    )
            if bias is not None:
                self.add_data_tensor(
                    "{layer_name}_bias".format(layer_name=layer_name),
                    bias
                    )
            
            self.add_output_tensor(layer_name, output_shape)
        elif target == torch.nn.functional.elu:
            self.model_forward += INDENT + "NN_elu(&model->{layer_name}, &model->{input_names[0]}, {eps});\n".format(
                layer_name=layer_name,
                input_names=input_names,
                eps=args[1]
            )
            self.add_output_tensor(layer_name, output_shape)

    def run_node(self, n: torch.fx.node.Node) -> Any:
        out = super().run_node(n)
        
        if n.op == "placeholder":
            print("placeholder:", n.name)
            
            ## sooooo hacky
            shape = self.example_input.shape
            if len(shape) == 4:
                shape = (shape[0], shape[2], shape[3], shape[1])
            
            self.model_struct += INDENT + TEMPLATE_TENSOR_DECLARE.format(name=n.name)
            
            self.model_init += INDENT + TEMPLATE_TENSOR_INIT.format(
                name=n.name,
                dim=len(shape),
                shape=", ".join(str(x) for x in shape),
                dtype=TorchConverter.dtype_to_str(self.example_input.dtype),
                data="NULL"
            )
            
        # elif n.op == "get_attr":
        #     breakpoint()
        
        elif n.op == "call_function":
            print("call function:", n.name, n.target)
            args = n.args
            
            if n.target == torch.nn.functional.linear:
                weight = self.model.state_dict()[n.args[1].target]
                bias = self.model.state_dict()[n.args[2].target]
                args = (n.args[0], weight, bias)
            
            self.trace_functional(n.name, n.target, args, n.kwargs, out)
        
        elif n.op == "call method":
            print("call method:", n.name, n.target)
            raise NotImplementedError()

        elif n.op == "call_module":
            print("call module:", n.name, n.target)
            args = n.args
            layer_name = n.name
            
            module = self.get_module(n.target)
            target = TorchConverter.to_functional_torch(module)
            
            self.model_init += "\n"
            self.model_init += INDENT + "// {module}: {layer_name}\n".format(
                    module=type(module),
                    layer_name=layer_name
                )

            if type(module) == torch.nn.Linear:
                args = (n.args[0], module.weight, module.bias)
                
            elif type(module) == torch.nn.BatchNorm2d:
                args = (n.args[0], module.weight, module.bias, module.running_mean, module.running_var, module.eps)
            
            elif type(module) == torch.nn.Conv2d:
                args = (n.args[0], module.weight, module.bias, module.stride, module.padding, module.dilation, module.groups)
            
            elif type(module) == torch.nn.ELU:
                args = (n.args[0], module.alpha)
            
            else:
                print("[WARNING] Unsupported module call:", n.target)
            
            self.trace_functional(layer_name, target, args, n.kwargs, out)
            
        return out
    
    def convert(self, example_input, output_dir="."):
        self.example_input = example_input
        
        # trace the model
        output = self.run(example_input)

        print("finished tracing the model")
        
        with open(os.path.join(output_dir, "model.h"), "w") as f:
            f.write(self.model_template.render(
                model_struct=self.model_struct, 
                model_init=self.model_init,
                model_forward=self.model_forward
            ))
        
        with open(os.path.join(output_dir, "model.bin"), "wb") as f:
            f.write(self.weight_content)
        
        return output


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)
        
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.seq = nn.Sequential(
                nn.Linear(48, 128, bias=True),
                nn.ELU(),
                nn.Linear(128, 5, bias=True),
            )
            self.lin2 = nn.Linear(5, 12, bias=True)

        def forward(self, input):
            x = self.seq.forward(input)
            x = F.relu(x)
            output = self.lin2.forward(x)
            return output

    m = Net()
    m.eval()

    test_input = torch.zeros((48, )).unsqueeze(0)
    print("input:", test_input)

    TorchConverter(m).print_graph()
    output = TorchConverter(m).convert(test_input)
    print("output:", output)
    
