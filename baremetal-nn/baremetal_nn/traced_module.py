import operator
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn
import torch.fx
import jinja2


class TracedModule(torch.fx.Interpreter):
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

    def __init__(self, model: torch.nn.Module):
        graph, gm = TracedModule._extract_graph_module(model)
        super().__init__(gm)

        # store the model, graph, and graph module as class attributes
        self.model: torch.nn.Module = model
        self.graph: torch.fx.Graph = graph
        self.gm: torch.fx.GraphModule = gm

        # extract node information
        # this dictionary maps each node name to a tuple of the node's args and kwargs
        # this is used for getting the input tensor and parameters for each forward function call
        self.node_info: Dict[str, Tuple[Any, Any]] = {n.name: (n.args, n.kwargs) for n in self.graph.nodes}

        # initialize jinja2 code generation environment
        self.env = jinja2.Environment()

        self.reset()
    
    def reset(self):
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

    def _get_inner_module(self, module: torch.nn.Module, target_hierarchy: List[str]) -> torch.nn.Module:
        """
        Get a module in a nn.Sequential layer.
        This function will recursively unpack the nn.Sequential layers to get the innermost module.
        
        Args:
            module (torch.nn.Sequential): A nn.Sequential layer.
            indicies (List[int]): A list of indicies of the layers in the nn.Sequential layer.
        
        Returns:
            The innermost module.
        """
        module_name = target_hierarchy[0]
        target_hierarchy = target_hierarchy[1:]
        submodule = getattr(module, module_name)

        if len(target_hierarchy) == 0:
            return submodule
        
        return self._get_inner_module(submodule, target_hierarchy)

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
            return self._get_inner_module(self.model, target_hierarchy)
        
        return getattr(self.model, module_name)

    def add_uninitialized_tensor(self, name: str, tensor: torch.Tensor):
        """
        Adds an uninitialized tensor to the C code.

        Args:
            name (str): The name of the tensor.
            tensor (torch.Tensor): The example tensor. This is used to determine the shape and data type of the tensor.
        """
        self.tensors[name] = {
            "tensor": tensor,
            "initialized": False
        }
    
    def add_initialized_tensor(self, name: str, tensor: torch.Tensor):
        """
        Adds an initialized tensor to the C code.

        Args:
            name (str): The name of the tensor.
            tensor (torch.Tensor): The tensor containing the data. This is used to determine the shape, data type, and data of the tensor.
        """
        self.tensors[name] = {
            "tensor": tensor,
            "initialized": True
        }

    def add_forward_call(self, function_name: str, out: torch.Tensor, layer_name: str, input_names: List[str], parameters: List[str] = None):
        """
        Adds a forward function call to the C code.

        Args:
            function_name (str): The name template of the function to call.
            out (torch.Tensor): The output tensor.
            layer_name (str): The name of the layer.
            input_names (List[str]): The names of the input tensors.
            parameters (List[str]): The additional parameters to pass to the function.
        """
        
        dtype_str = TracedModule.get_dtype_str(out.dtype)
        
        # get the nn function name and format it
        function_name = function_name.format(
            dim=out.dim(),
            dtype=dtype_str.lower()
            )
        
        # get the argument list
        args_list = [layer_name] + input_names  # output tensor is the same as layer name
        args_list = [f"&model->{arg_name}" if arg_name is not None else "NULL" for arg_name in args_list]
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

        n has the following attributes:
         - op: the operation that is being performed (here it is "call_function")
         - name: the name of the layer (e.g. "relu", "linear")
         - target: the Python function that is being called (e.g. torch.nn.functional.relu, torch.nn.functional.linear)
         - args: a list of torch.fx.node.Node objects that are the arguments to the function
         - prev: the previous nodes in the graph
         - next: the next nodes in the graph
        """
        print("call function:", n.name, n.target, n.args)

        # get all the related information
        function = n.target
        layer_name = n.name
        input_names = []
        input_args = n.args

        for n in self.node_info[n.name][0]:
            input_name = n.name if n is not None else None
            input_names.append(input_name)
        
        # Math operations - Pointwise Ops
        if function == operator.__add__:
            self.add_uninitialized_tensor(layer_name, out)
            self.add_forward_call("nn_add{dim}d_{dtype}", out, layer_name, input_names)
        
        elif function == operator.__mul__:
            self.add_uninitialized_tensor(layer_name, out)
            self.add_forward_call("nn_mul{dim}d_{dtype}", out, layer_name, input_names)
        
        # Convolution Layers

        # Non-linear Activations
        elif function == torch.nn.functional.relu:
            self.add_uninitialized_tensor(layer_name, out)
            self.add_forward_call("nn_relu{dim}d_{dtype}", out, layer_name, input_names)
        
        elif function == torch.nn.functional.relu6:
            self.add_uninitialized_tensor(layer_name, out)
            self.add_forward_call("nn_relu6{dim}d_{dtype}", out, layer_name, input_names)

        elif function == torch.nn.functional.softmax:
            self.add_uninitialized_tensor(layer_name, out)
            self.add_forward_call("nn_softmax{dim}d_{dtype}", out, layer_name, input_names)
        
        elif function == torch.nn.functional.tanh:
            self.add_uninitialized_tensor(layer_name, out)
            self.add_forward_call("nn_tanh{dim}d_{dtype}", out, layer_name, input_names)
        
        # Linear Layers
        elif function == torch.nn.functional.linear:
            weight = self.model.state_dict()[input_args[1].target]
            if input_args[2] is not None:
                bias = self.model.state_dict()[input_args[2].target]
            else:
                bias = None
            self.add_uninitialized_tensor(layer_name, out)
            self.add_initialized_tensor(f"{input_names[1]}", weight)
            if bias is not None:
                self.add_initialized_tensor(f"{input_names[2]}", bias)
            self.add_forward_call("nn_linear_{dtype}", out, layer_name, input_names)
        
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

        elif type(module) == torch.nn.Softmax:
            self.add_uninitialized_tensor(layer_name, out)
            self.add_forward_call("nn_softmax{dim}d_{dtype}", out, layer_name, input_names)
        
        elif type(module) == torch.nn.Tanh:
            self.add_uninitialized_tensor(layer_name, out)
            self.add_forward_call("nn_tanh{dim}d_{dtype}", out, layer_name, input_names)
    
        # Linear Layers
        elif type(module) == torch.nn.Linear:
            weight = module.weight
            # optionally use the bias if it exists
            bias = module.bias if module.bias is not None else None

            input_names.append(f"{layer_name}_weight")
            if bias is not None:
                input_names.append(f"{layer_name}_bias")
            else:
                input_names.append(None)

            self.add_uninitialized_tensor(layer_name, out)
            self.add_initialized_tensor(f"{layer_name}_weight", weight)
            if bias is not None:
                self.add_initialized_tensor(f"{layer_name}_bias", bias)
            self.add_forward_call("nn_linear_{dtype}", out, layer_name, input_names)
        
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

    def convert(self, output_directory: str = "./", model_name: str = "model"):
        """
        Convert the model to a C model.

        Args:
            args: The input to the model.
            kwargs: The keyword arguments to the model.
        
        Returns:
            The output of the model.
        """
        if self.example_inputs is None:
            raise ValueError("No example inputs provided. Please call forward() at least once.")

        # === Generate the tensor structs and initialize routines for the tensors in the C code. ===
        for name, tensor in self.tensors.items():
            initialized = tensor["initialized"]
            tensor = tensor["tensor"]
            
            dim = tensor.dim()
            dtype_str = TracedModule.get_dtype_str(tensor.dtype)
            self.model_struct.append(f"Tensor{dim}D_{dtype_str} {name};")
            
            for i in range(tensor.dim()):
                self.model_init.append(f"model->{name}.shape[{i}] = {tensor.shape[i]};")
            
            if initialized:
                self.model_init.append(f"model->{name}.data = (float *)(model_weight_data + {len(self.weight_content)});")    
                self.weight_content += tensor.detach().numpy().tobytes()
            else:
                n_size = tensor.nelement() * tensor.element_size()
                self.model_init.append(f"model->{name}.data = (float *)malloc({n_size});")


        print("finished tracing the model")

        # === Write the generated C code to the output directory. ===
        # create the output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        INDENT = "  "
        model_struct = [f"{INDENT}{line}" for line in self.model_struct]
        model_init = [f"{INDENT}{line}" for line in self.model_init]
        model_forward = [f"{INDENT}{line}" for line in self.model_forward]

        model_struct_str = "\n".join(model_struct)
        model_init_str = "\n".join(model_init)
        model_forward_str = "\n".join(model_forward)

        model_h_path = os.path.join(output_directory, f"{model_name}.h")
        model_bin_path = os.path.join(output_directory, f"{model_name}.bin")

        with open(model_h_path, "w") as f:
            f.write(TracedModule.MODEL_H_TEMPLATE.format(
                model_struct=model_struct_str,
                model_init=model_init_str,
                model_forward=model_forward_str
            ))
        
        with open(model_bin_path, "wb") as f:
            f.write(self.weight_content)
        
        print(f"wrote the model to {model_h_path} and {model_bin_path}")

    def forward(self, *args):
        self.reset()
        self.example_inputs = args

        output = self.run(*args)

        return output
    
    def __call__(self, *args):
        return self.forward(*args)


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # set the seed for reproducibility
    torch.manual_seed(0)
    
    # define a simple model
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.lin1 = nn.Linear(48, 256, bias=False)
            self.lin2 = nn.Linear(256, 128, bias=False)
            self.lin3 = nn.Linear(128, 5, bias=False)

        def forward(self, input):
            x = self.lin1.forward(input)
            x = F.relu(x)
            x = self.lin2.forward(x)
            x = F.relu(x)
            x = self.lin3.forward(x)
            return x

    # create the model
    m = Net()

    # set the model to evaluation mode
    m.eval()

    # create an example input
    example_input = torch.ones((48, )).unsqueeze(0)
    print("input:")
    print(example_input)

    # trace the model
    m = TracedModule(m)

    # print the model graph
    m.print_graph()

    # forward the model to get the shape of each layer
    # a.k.a. trace the model
    output = m.forward(example_input)
    print("output:")
    print(output)

    # convert the model to a C model
    # this function will create the model.h and model.bin files
    # under the current execution directory
    m.convert(
        output_directory="./",
        model_name="model"
    )

    