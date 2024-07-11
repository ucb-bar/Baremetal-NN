import random

import torch
import jinja2


# see if we have a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))


# seed
seed = 0

random.seed(seed)
torch.manual_seed(seed)

env = jinja2.Environment()


env.globals["len"] = len
env.globals["type"] = type
env.globals["str"] = str
env.globals["torch"] = torch


def rand(shape):
    return (torch.rand(shape, dtype=torch.float32, device=device) - 0.5) * 10

def rand16(shape):
    return (torch.rand(shape, dtype=torch.float16, device=device) - 0.5) * 2


test_pattern = [
    # ("abs",         lambda a: torch.abs(a),             [("a", rand((7, 7))),                                           ]),
    # ("add",         lambda a, b: a + b,                 [("a", rand((6, 7))),         ("b", rand((6, 7)))               ]),
    # ("add",         lambda a, b: a + b,                 [("a", rand((6, 7))),         ("b", rand((1, 7)))               ]),
    # ("add",         lambda a, b: a + b,                 [("a", rand((6, 7))),         ("b", rand((6, 1)))               ]),
    # ("add",         lambda a, b: a + b,                 [("a", rand((6, 7))),         ("b", rand((7, )))                ]),
    # ("add_inplace", lambda a, b: a + b,                 [("actual", torch.zeros((7, 7))),   ("b", rand((7, 7)))         ]),
    # ("add1",        lambda a, b: a + b,                 [("a", rand((7, 7))),         ("v", random.random())            ]),
    # ("clip",        lambda a, v_min, v_max: torch.clip(a, v_min, v_max),  
    #     [("a", rand((7, 7))), ("v_min", random.random() - 1), ("v_max", random.random())                                ]),
    # ("div",         lambda a, b: a / b,                 [("a", rand((7, 7))),         ("b", rand((7, 7)))               ]),
    # ("fill",        lambda a, v: a.fill_(v),            [("actual", torch.zeros((7, 7))),   ("v", random.random())      ]),
    # ("matmul_t",    lambda a, b: a @ b.T,               [("a", rand((6, 7))),         ("b", rand((5, 7)))               ]),
    # ("matmul",      lambda a, b: a @ b,                 [("a", rand((6, 7))),         ("b", rand((7, 5)))               ]),
    # ("max",         lambda a: torch.max(a),             [("a", rand((7, 7)))                                            ]),
    # ("maximum",     lambda a, b: torch.maximum(a, b),   [("a", rand((7, 7))),         ("b", rand((7, 7)))               ]),
    # ("min",         lambda a: torch.min(a),             [("a", rand((7, 7)))                                            ]),
    # ("minimum",     lambda a, b: torch.minimum(a, b),   [("a", rand((7, 7))),         ("b", rand((7, 7)))               ]),
    # ("mul",         lambda a, b: a * b,                 [("a", rand((7, 7))),         ("b", rand((7, 7)))               ]),
    # ("mul1",        lambda a, b: a * b,                 [("a", rand((7, 7))),         ("v", random.random())            ]),
    # ("neg",         lambda a: -a,                       [("a", rand((7, 7))),                                           ]),
    # ("sub",         lambda a, b: a - b,                 [("a", rand((7, 7))),         ("b", rand((7, 7)))               ]),
    # ("sum",         lambda a: torch.sum(a),             [("a", rand((7, 7))),                                           ]),
    
    # ("linear",      lambda x, w, b: torch.nn.functional.linear(x, w, b), 
    #     [("x", rand((6, 7))), ("w", rand((5, 7))), ("b", rand((1, 5)))                                                  ]),
    # ("linear",      lambda x, w, b: torch.nn.functional.linear(x, w, b),
    #     [("x", rand((6, 7))), ("w", rand((5, 7))), ("b", rand((5, )))                                                   ]),
    # ("relu",        lambda x: torch.nn.functional.relu(x),
    #     [("x", rand((7, 7)))                                                                                            ]),
    # ("softmax",     lambda a: torch.nn.functional.softmax(a, dim=0),
    #     [("x", rand((7, 7))+1), ("0", None)                                                                             ]),
    # ("softmax",     lambda a: torch.nn.functional.softmax(a, dim=1),
    #     [("x", rand((7, 7))+1), ("1", None)                                                                             ]),
    # ("softmax",     lambda a: torch.nn.functional.softmax(a, dim=-1),
    #     [("x", rand((7, 7))+1), ("-1", None)                                                                            ]),
    # ("relu6",       lambda x: torch.nn.functional.relu6(x),    
    #     [("x", rand((7, 7)))                                                                                            ]),
    # ("conv2d",      lambda x, w, b: torch.nn.functional.conv2d(x.permute((0, 3, 1, 2)), w.permute((3, 2, 0, 1)), b, stride=1, padding=0, dilation=1, groups=1).permute((0, 2, 3, 1)),
    #     [("x", rand((1, 16, 16, 3))), ("w", rand((3, 3, 3, 6))), ("b", rand((6, ))), 
    #      ("(size_t[]){1, 1}, (size_t[]){0, 0}, (size_t[]){1, 1}, 1", None)                                              ]),
    # ("conv2d",      lambda x, w, b: torch.nn.functional.conv2d(x.permute((0, 3, 1, 2)), w.permute((3, 2, 0, 1)), b, stride=1, padding=1, dilation=1, groups=1).permute((0, 2, 3, 1)),
    #     [("x", rand((1, 16, 16, 3))), ("w", rand((3, 3, 3, 71))), ("b", rand((71, ))),
    #      ("(size_t[]){1, 1}, (size_t[]){1, 1}, (size_t[]){1, 1}, 1", None)                                              ]),
    # ("nchw_to_nhwc",  lambda x: x.permute((0, 2, 3, 1)),  [("x", rand((1, 2, 3, 3)))                                    ]),
    # ("nhwc_to_nchw",  lambda x: x.permute((0, 3, 1, 2)),  [("x", rand((1, 3, 3, 2)))                                    ]),
    # ("conv2d",      lambda x, w, b: torch.nn.functional.conv2d(x.permute((0, 3, 1, 2)), w.permute((3, 2, 0, 1)), b, stride=1, padding=1, dilation=1, groups=16).permute((0, 2, 3, 1)),
    #     [("x", rand((1, 12, 12, 16))), ("w", rand((3, 3, 1, 16))), ("b", rand((16, ))),
    #      ("(size_t[]){1, 1}, (size_t[]){1, 1}, (size_t[]){1, 1}, 16", None)                                             ]),
    # ("conv2d",      lambda x, w, b: torch.nn.functional.conv2d(x.permute((0, 3, 1, 2)), w.permute((3, 2, 0, 1)), b, stride=1, padding=1, dilation=1, groups=1).permute((0, 2, 3, 1)),
    #     [("x", rand((1, 12, 12, 16))), ("w", rand((3, 3, 16, 56))), ("b", rand((56, ))), 
    #      ("(size_t[]){1, 1}, (size_t[]){1, 1}, (size_t[]){1, 1}, 1", None)                                              ]),
    
    # ("layer_norm",   lambda x, w, b: torch.nn.functional.layer_norm(x, (x.shape[1], ), w, b, eps=1e-05), 
    #     [("x", rand((6, 5))), ("1", None), ("w", rand((5))), ("b", torch.zeros((5))), ("1e-05", None)         ]),
    # ("layer_norm",   lambda x, w, b: torch.nn.functional.layer_norm(x, (x.shape[1], ), w, b, eps=1e-05), 
    #     [("x", rand((6, 5))), ("1", None), ("w", rand((5))), ("b", rand((5))), ("1e-05", None)                      ]),

    ("abs",         lambda a: torch.abs(a),             [("a", rand16((1, 7))),                                         ]),
    ("add",         lambda a, b: a + b,                 [("a", rand16((6, 7))),       ("b", rand16((6, 7)))             ]),
    ("matmul_t",    lambda a, b: a @ b.T,               [("a", rand16((6, 7))),       ("b", rand16((5, 7)))             ]),
    ("matmul",      lambda a, b: a @ b,                 [("a", rand16((6, 7))),       ("b", rand16((7, 5)))             ]),
    
    ("linear",      lambda x, w, b: torch.nn.functional.linear(x, w, b), 
        [("x", rand16((6, 7))), ("w", rand16((5, 7))), ("b", rand16((1, 5)))                                            ]),
    ("relu",        lambda x: torch.nn.functional.relu(x),
        [("x", rand16((7, 7)))                                                                                          ]),
    
]


# NN_printf(golden);
# NN_printf(actual);


c_code = """
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <rv.h>

#include "nn.h"
#include "unittest.h"

int main() {
  enable_accelerator_features();

  size_t cycles = 0;

  {{ code }}

}
"""


def type_to_str(dtype: torch.dtype):
    if dtype == torch.float16:
        return "DTYPE_F16"
    elif dtype == torch.float32:
        return "DTYPE_F32"



def format_tensor(name: str, tensor: torch.Tensor):
    dim = len(tensor.shape)
    shape = ", ".join([str(s) for s in tensor.shape])
    dtype = type_to_str(tensor.dtype)
    data_np = tensor.cpu().contiguous().numpy()
    data = ",".join([hex(b) for b in data_np.flatten().tobytes()])
    human_readable = str(data_np).replace("\n", " ")[:80]
    tensor_str = env.from_string("""
    // {{ human_readable }}
    Tensor *{{ name }} = NN_tensor({{ dim }}, (size_t[]){ {{ shape }} }, {{ dtype }}, (uint8_t[]){ {{ data }} });""").render(
        human_readable=human_readable, name=name, dim=dim, shape=shape, dtype=dtype, data=data
    )
    return tensor_str
    


def generate_test_pattern(op, function, inputs):
    actual_inputs = [value for name, value in inputs if value is not None]
    result = function(*actual_inputs)


    tensor_constructors = []
    tensor_destructors = []

    for name, value in inputs:
        if type(value) == str:
            pass
        
        if type(value) == torch.Tensor and name != "actual":
            dim = len(value.shape)
            shape = ", ".join([str(s) for s in value.shape])
            dtype = type_to_str(value.dtype)
            data_np = value.cpu().contiguous().numpy()
            data = ", ".join([str(b) for b in data_np.flatten().tobytes()])

            human_readable = str(data_np).replace("\n", " ")[:80]
            tensor_str = format_tensor(name, value)
            
            tensor_constructors.append(tensor_str)
            tensor_destructors.append("    NN_delete_tensor({});\n".format(name))

        elif type(value) == float:
            tensor_str = env.from_string("float {{ name }} = {{ value }};").render(name=name, value=value)
            tensor_constructors.append(tensor_str)
        

    golden_str = format_tensor("golden", result)

    dim = len(result.shape)
    shape = ", ".join([str(s) for s in result.shape])
    dtype = type_to_str(result.dtype)
    
    result_tensors = golden_str + env.from_string("""
    Tensor *actual = NN_zeros({{ dim }}, (size_t[]){ {{ shape }} }, {{ dtype }});""").render(
        dim=dim, shape=shape, dtype=dtype
    )

    inputs = ", ".join([name for name, value in inputs if name != "actual"])
    inputs = ", " + inputs if inputs else inputs

    func_str = env.from_string("""    NN_{{ op }}(actual{{ inputs }});\n""").render(
        op=op, inputs=inputs
    )
    
    precision = "1e-4"
    if result.dtype == torch.float16:
        precision = "1e-2"

    test_template = env.from_string("""
  {
    printf("{{ (op + ":").ljust(24) }}");
{% for tensor_str in tensor_constructors %}{{ tensor_str }}{% endfor %}

{{ result_tensors }}

    cycles = read_cycles();
{{ func_str }}                          
    cycles = read_cycles() - cycles;
    printf("%s  (%lu cycles)\\n", compare_tensor(golden, actual, {{ precision }}) ? "PASS" : "FAIL", cycles);

{% for tensor_str in tensor_destructors %}{{ tensor_str }}{% endfor %}
    NN_delete_tensor(golden);
    NN_free_tensor_data(actual);
    NN_delete_tensor(actual);
  }
""")
    
    return test_template.render(op=op, tensor_constructors=tensor_constructors, tensor_destructors=tensor_destructors, result_tensors=result_tensors, func_str=func_str, precision=precision)


template = env.from_string(c_code)



result = template.render(code="\n".join([generate_test_pattern(*pattern) for pattern in test_pattern]))

with open("generated.c", "w") as f:
    f.write(result)

