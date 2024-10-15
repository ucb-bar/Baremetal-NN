import operator
import os
import inspect
from typing import Any, Dict, List, Tuple, Callable

import numpy as np
import torch
import torch.nn
import torch.fx
import jinja2
import tabulate




class TorchConverter(torch.fx.Interpreter):
    """
    This class converts a PyTorch model to a C model.
    """

    MODEL_H_TEMPLATE = """#ifndef __MODEL_H
#define __MODEL_H

#include "nn.h"

// load the weight data block from the model.bin file
INCLUDE_FILE(".rodata", "./model.bin", model_weight);
extern uint8_t model_weight_data[];
extern size_t model_weight_start[];
extern size_t model_weight_end[];

typedef struct {{
{model_struct}
}} Model;

void model_init(Model* model) {{
{model_init}
}}

void model_forward(Model* model) {{
{model_forward}
}}

#endif  // __MODEL_H
"""
    @staticmethod
    def get_dtype_str(dtype: torch.dtype) -> str:
        """
        Convert a PyTorch dtype to a string representing the NN type.

        Args:
            dtype (torch.dtype): The PyTorch dtype to convert.
        
        Returns:
            str: The string representing the NN type.
        """
        if dtype == torch.float16:
            return "F16"
        elif dtype == torch.float32:
            return "F32"
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
    @staticmethod
    def get_ctype_str(dtype: torch.dtype) -> str:
        """
        Convert a PyTorch dtype to a string representing the C type.

        Args:
            dtype (torch.dtype): The PyTorch dtype to convert.
        
        Returns:
            str: The string representing the C type.
        """
        if dtype == torch.float16:
            return "float16_t"
        elif dtype == torch.float32:
            return "float32"
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    @staticmethod
    def _extract_graph_module(model: torch.nn.Module) -> tuple[torch.fx.Graph, torch.fx.GraphModule]:
        """
        Helper function to extract the graph module from the model.

        Args:
            model (torch.nn.Module): The model to extract the graph module from.
        
        Returns:
            tuple[torch.fx.Graph, torch.fx.GraphModule]: The graph and the graph module.
        """
        graph = torch.fx.Tracer().trace(model)
        # Does some checks to make sure the Graph is well-formed.
        graph.lint()
        gm = torch.fx.GraphModule(model, graph)
        return graph, gm

    def __init__(self, model: torch.nn.Module, output_directory: str = "./"):
        graph, gm = TorchConverter._extract_graph_module(model)
        super().__init__(gm)

        # store the model, graph, and graph module as class attributes
        self.model: torch.nn.Module = model
        self.graph: torch.fx.Graph = graph
        self.gm: torch.fx.GraphModule = gm

        self.output_directory = output_directory

        # extract node information
        self.node_info: Dict[str, Tuple[Any, Any]] = {n.name: (n.args, n.kwargs) for n in self.graph.nodes}

        # initialize jinja2 code generation environment
        self.env = jinja2.Environment()

        # arrays to hold the to-be-generated code
        self.model_struct = []
        self.model_init = []
        self.model_forward = []
        self.weight_content = b""

        # dictionaries to hold the tensor data
        self.tensors = {}

        # this is sooooo hacky
        self.placeholder_counter: Dict[str, int] = {}
        self.function_counter: Dict[str, int] = {}

    def print_graph(self):
        """
        Print the graph in a tabular format in the terminal.
        """
        self.gm.graph.print_tabular()

    def get_modules_from_sequential(self, module: torch.nn.Sequential, indicies: List[int]) -> torch.nn.Module:
        """
        Get a module in a nn.Sequential layer.
        This function will recursively unpack the nn.Sequential layers to get the innermost module.
        
        Args:
            module (torch.nn.Sequential): A nn.Sequential layer.
            indicies (List[int]): A list of indicies of the layers in the nn.Sequential layer.
        
        Returns:
            The innermost module.
        """
        if len(indicies) == 0:
            return module
        return self.get_modules_from_sequential(module[indicies[0]], indicies[1:])

    def get_module(self, module_name: str) -> torch.nn.Module:
        """
        Finds the module specified by the module name from the model.
        If the module name contains a dot, it will recursively unpack the nn.Sequential layers to get the innermost module.
        
        Args:
            module_name (str): The name of the module to get.
        
        Returns:
            The target module.
        """
        if "." in module_name:
            # if we have nn.Sequential layers
            target_hierarchy = module_name.split(".")
            sequential_name = target_hierarchy[0]

            # indicies = target_hierarchy[1:]
            indicies = [int(x) for x in target_hierarchy[1:]]

            module = getattr(self.model, sequential_name)
            return self.get_modules_from_sequential(module, indicies)
        
        return getattr(self.model, module_name)

    def add_uninitialized_tensor(self, name: str, tensor: torch.Tensor):
        """
        Add an uninitialized tensor to the C code.
        """
        self.tensors[name] = {
            "tensor": tensor,
            "initialized": False
        }
    
    def add_initialized_tensor(self, name: str, tensor: torch.Tensor):
        """
        Add an initialized tensor to the C code.
        """
        self.tensors[name] = {
            "tensor": tensor,
            "initialized": True
        }

    def add_forward_call(self, function_name: str, out: torch.Tensor, layer_name: str, input_names: List[str], parameters: List[str] = None):
        """
        This method creates the C code for the forward call.

        Args:
            function (Callable): The function to call.
            dim (int): The dimension of the output tensor.
            dtype (torch.dtype): The data type of the output tensor.
            layer_name (str): The name of the layer.
            input_names (List[str]): The names of the input tensors.
        """
        
        dtype_str = TorchConverter.get_dtype_str(out.dtype)
        
        # get the nn function name and format it
        function_name = function_name.format(
            dim=out.dim(),
            dtype=dtype_str.lower()
            )
        
        # get the argument list
        args_list = [layer_name] + input_names  # output tensor is the same as layer name
        args_list = [f"&model->{arg_name}" for arg_name in args_list]
        if parameters:
            args_list += [str(param) for param in parameters]
        arg_list_str = ", ".join(args_list)
    
        self.model_forward.append(f"{function_name}({arg_list_str});")
    

    def handle_placeholder(self, n: torch.fx.node.Node, out: torch.Tensor):
        print("placeholder:", n.name)
        self.add_uninitialized_tensor(n.name, out)

    def handle_get_attr(self, n: torch.fx.node.Node, out: torch.Tensor):
        # print("get attr:", n.name, n.target)
        pass

    def handle_call_function(self, n: torch.fx.node.Node, out: torch.Tensor):
        """
        Handle the case where the node is a call to a torch function (e.g. relu, elu, etc.)
        """
        print("call function:", n.name, n.target, n.args)

        # get all the related information
        function = n.target
        layer_name = n.name
        input_names = [n.name for n in self.node_info[n.name][0]]
        input_args = n.args
        
        # Math operations - Pointwise Ops
        if function == operator.__add__:
            self.add_uninitialized_tensor(layer_name, out)
            self.add_forward_call("nn_add_{dtype}", out, layer_name, input_names)
        
        elif function == operator.__mul__:
            self.add_uninitialized_tensor(layer_name, out)
            self.add_forward_call("nn_mul_{dtype}", out, layer_name, input_names)
        
        # Convolution Layers

        # Non-linear Activations
        elif function == torch.nn.functional.relu:
            self.add_uninitialized_tensor(layer_name, out)
            self.add_forward_call("nn_relu{dim}d_{dtype}", out, layer_name, input_names)
        
        elif function == torch.nn.functional.relu6:
            self.add_uninitialized_tensor(layer_name, out)
            self.add_forward_call("nn_relu6{dim}d_{dtype}", out, layer_name, input_names)
        
        elif function == torch.nn.functional.tanh:
            self.add_uninitialized_tensor(layer_name, out)
            self.add_forward_call("nn_tanh{dim}d_{dtype}", out, layer_name, input_names)
        
        # Linear Layers
        elif function == torch.nn.functional.linear:
            weight = self.model.state_dict()[input_args[1].target]
            bias = self.model.state_dict()[input_args[2].target]
            self.add_uninitialized_tensor(layer_name, out)
            self.add_initialized_tensor(f"{input_names[1]}", weight)
            self.add_initialized_tensor(f"{input_names[2]}", bias)
            self.add_forward_call("nn_addmm_{dtype}", out, layer_name, input_names)
        
        # Vision Functions

        
    def handle_call_method(self, n: torch.fx.node.Node, out: torch.Tensor):
        print("call method:", n.name, n.target)
        raise NotImplementedError()

    def handle_call_module(self, n: torch.fx.node.Node, out: torch.Tensor):
        print("call module:", n.name, n.target)

        module = self.get_module(n.target)
        layer_name = n.name
        input_names = [n.name for n in self.node_info[n.name][0]]

        # Convolution Layers
        if type(module) == torch.nn.Conv2d:
            self.add_uninitialized_tensor(layer_name, out)
            self.add_initialized_tensor(f"{layer_name}_weight", module.weight)
            self.add_initialized_tensor(f"{layer_name}_bias", module.bias)
            self.add_forward_call("nn_conv2d_{dtype}", out, layer_name, input_names, [
                module.stride,
                module.padding,
                module.dilation,
                module.groups
            ])

        # Normalization Layers
        elif type(module) == torch.nn.BatchNorm2d:
            self.add_uninitialized_tensor(layer_name, out)
            self.add_initialized_tensor(f"{layer_name}_weight", module.weight)
            self.add_initialized_tensor(f"{layer_name}_bias", module.bias)
            self.add_initialized_tensor(f"{layer_name}_running_mean", module.running_mean)
            self.add_initialized_tensor(f"{layer_name}_running_var", module.running_var)
            self.add_forward_call("nn_batchnorm2d_{dtype}", out, layer_name, input_names, [module.eps])
        
        # Non-linear Activations
        elif type(module) == torch.nn.ELU:
            self.add_uninitialized_tensor(layer_name, out)
            self.add_forward_call("nn_elu{dim}d_{dtype}", out, layer_name, input_names, [module.alpha])
        
        elif type(module) == torch.nn.ReLU:
            self.add_uninitialized_tensor(layer_name, out)
            self.add_forward_call("nn_relu{dim}d_{dtype}", out, layer_name, input_names)
        
        elif type(module) == torch.nn.ReLU6:
            self.add_uninitialized_tensor(layer_name, out)
            self.add_forward_call("nn_relu6{dim}d_{dtype}", out, layer_name, input_names)
        
        elif type(module) == torch.nn.Tanh:
            self.add_uninitialized_tensor(layer_name, out)
            self.add_forward_call("nn_tanh{dim}d_{dtype}", out, layer_name, input_names)
    
        # Linear Layers
        elif type(module) == torch.nn.Linear:
            weight = module.weight
            bias = module.bias
            input_names.append(f"{layer_name}_weight")
            input_names.append(f"{layer_name}_bias")

            self.add_uninitialized_tensor(layer_name, out)
            self.add_initialized_tensor(f"{layer_name}_weight", weight)
            self.add_initialized_tensor(f"{layer_name}_bias", bias)
            self.add_forward_call("nn_addmm_{dtype}", out, layer_name, input_names)
        
    def handle_output(self, n: torch.fx.node.Node, out: torch.Tensor):
        print("output:", n.name, out.shape, out.dtype)
        n_size = out.nelement() * out.element_size()
        
        self.add_uninitialized_tensor(n.name, out)
        self.model_forward.append(f"memcpy(model->output.data, model->{n.args[0].name}.data, {n_size});")
        
    def run_node(self, n: torch.fx.node.Node) -> torch.Tensor:
        out = super().run_node(n)

        if n.op == "placeholder":
            self.handle_placeholder(n, out)
        elif n.op == "get_attr":
            self.handle_get_attr(n, out)
        elif n.op == "call_function":
            self.handle_call_function(n, out)
        elif n.op == "call_method":
            self.handle_call_method(n, out)
        elif n.op == "call_module":
            self.handle_call_module(n, out)
        elif n.op == "output":
            self.handle_output(n, out)

        return out

    def _generate_tensors(self):
        """
        Generate the tensor structs and initialize routines for the tensors in the C code.
        """
        for name, tensor in self.tensors.items():
            initialized = tensor["initialized"]
            tensor = tensor["tensor"]
            
            dim = tensor.dim()
            dtype_str = TorchConverter.get_dtype_str(tensor.dtype)
            self.model_struct.append(f"Tensor{dim}D_{dtype_str} {name};")
            
            for i in range(tensor.dim()):
                self.model_init.append(f"model->{name}.shape[{i}] = {tensor.shape[i]};")
            
            if initialized:
                self.model_init.append(f"model->{name}.data = (float *)(model_weight_data + {len(self.weight_content)});")    
                self.weight_content += tensor.detach().numpy().tobytes()
            else:
                n_size = tensor.nelement() * tensor.element_size()
                self.model_init.append(f"model->{name}.data = (float *)malloc({n_size});")

    def _write_to_file(self):
        """
        Write the generated C code to the output directory.

        Args:
            output_directory (str): The directory to write the C code to.
        """
        # create the output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)

        INDENT = "  "
        model_struct = [f"{INDENT}{line}" for line in self.model_struct]
        model_init = [f"{INDENT}{line}" for line in self.model_init]
        model_forward = [f"{INDENT}{line}" for line in self.model_forward]

        model_struct_str = "\n".join(model_struct)
        model_init_str = "\n".join(model_init)
        model_forward_str = "\n".join(model_forward)

        with open(os.path.join(self.output_directory, "model.h"), "w") as f:
            f.write(TorchConverter.MODEL_H_TEMPLATE.format(
                model_struct=model_struct_str,
                model_init=model_init_str,
                model_forward=model_forward_str
            ))
        
        with open(os.path.join(self.output_directory, "model.bin"), "wb") as f:
            f.write(self.weight_content)

    def convert(self, *args, **kwargs):
        """
        Convert the model to a C model.

        Args:
            args: The input to the model.
            kwargs: The keyword arguments to the model.
        
        Returns:
            The output of the model.
        """
        self.example_inputs = args

        # trace the model
        output = self.run(*args)

        self._generate_tensors()
        print("finished tracing the model")

        self._write_to_file()

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
            x = F.relu(x)
            return output

    m = Net()
    m.eval()

    example_input = torch.ones((48, )).unsqueeze(0)
    print("input:")
    print(example_input)

    TorchConverter(m).print_graph()
    output = TorchConverter(m, "./examples/mlp").convert(example_input)
    print("output:")
    print(output)
    