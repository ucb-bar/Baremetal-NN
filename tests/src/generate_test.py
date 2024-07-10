import random

import torch
import jinja2

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
    return (torch.rand(shape, dtype=torch.float32) - 0.5) * 10

def rand16(shape):
    return (torch.rand(shape, dtype=torch.float16) - 0.5) * 10


test_pattern = [
    # ("abs",         lambda a: torch.abs(a),             [("a", rand((7, 7))),                                           ]),
    # ("add",         lambda a, b: a + b,                 [("a", rand((6, 7))),         ("b", rand((6, 7)))               ]),
    # ("add",         lambda a, b: a + b,                 [("a", rand((6, 7))),         ("b", rand((1, 7)))               ]),
    # ("add",         lambda a, b: a + b,                 [("a", rand((6, 7))),         ("b", rand((6, 1)))               ]),
    # ("add",         lambda a, b: a + b,                 [("a", rand((6, 7))),         ("b", rand((7, )))               ]),
    # ("addInplace",  lambda a, b: a + b,                 [("actual", torch.zeros((7, 7))),   ("b", rand((7, 7)))         ]),
    # ("add1",        lambda a, b: a + b,                 [("a", rand((7, 7))),         ("v", random.random())            ]),
    # ("clip",        lambda a, v_min, v_max: torch.clip(a, v_min, v_max),  [("a", rand((7, 7))), ("v_min", random.random() - 1), ("v_max", random.random())]),
    # ("div",         lambda a, b: a / b,                 [("a", rand((7, 7))),         ("b", rand((7, 7)))               ]),
    # ("fill",        lambda a, v: a.fill_(v),            [("actual", torch.zeros((7, 7))),   ("v", random.random())      ]),
    # ("matmulT",     lambda a, b: a @ b.T,               [("a", rand((6, 7))),         ("b", rand((5, 7)))               ]),
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
    
    # ("Linear",      lambda x, w, b: torch.nn.functional.linear(x, w, b), 
    #     [("x", rand((6, 7))), ("w", rand((5, 7))), ("b", rand((1, 5)))                                                 ]),
    # ("Linear",      lambda x, w, b: torch.nn.functional.linear(x, w, b),
    #     [("x", rand((6, 7))), ("w", rand((5, 7))), ("b", rand((5, )))                                                  ]),
    # ("ReLU",        lambda x: torch.nn.functional.relu(x),
    #     [("x", rand((7, 7)))                                                                                            ]),
    # ("Softmax",     lambda a: torch.nn.functional.softmax(a, dim=0),
    #     [("x", torch.ones((7, 7)))  ],
    #     ", 0"                                                                                                            ),
    # ("Softmax",     lambda a: torch.nn.functional.softmax(a, dim=1),
    #     [("x", torch.ones((7, 7)))  ],
    #     ", 1"                                                                                                            ),
    # ("ReLU6",       lambda x: torch.nn.functional.relu6(x),    
    #     [("x", rand((7, 7)))                                                                                            ]),
    # ("Conv2d",      lambda x, w, b: torch.nn.functional.conv2d(x.permute((0, 3, 1, 2)), w.permute((3, 2, 0, 1)), b, stride=1, padding=0, dilation=1, groups=1).permute((0, 2, 3, 1)),
    #     [("x", rand((1, 16, 16, 3))), ("w", rand((3, 3, 3, 6))), ("b", rand((6, )))],
    #     ", (size_t[]){1, 1}, (size_t[]){0, 0}, (size_t[]){1, 1}, 1"                                                      ),
    # ("Conv2d",      lambda x, w, b: torch.nn.functional.conv2d(x.permute((0, 3, 1, 2)), w.permute((3, 2, 0, 1)), b, stride=1, padding=1, dilation=1, groups=1).permute((0, 2, 3, 1)),
    #     [("x", rand((1, 16, 16, 3))), ("w", rand((3, 3, 3, 71))), ("b", rand((71, )))],
    #     ", (size_t[]){1, 1}, (size_t[]){1, 1}, (size_t[]){1, 1}, 1"                                                      ),
    # ("NCHWToNHWC",  lambda x: x.permute((0, 2, 3, 1)),  [("x", rand((1, 2, 3, 3)))                                     ]),
    # ("NHWCToNCHW",  lambda x: x.permute((0, 3, 1, 2)),  [("x", rand((1, 3, 3, 2)))                                     ]),
    # ("Conv2d",      lambda x, w, b: torch.nn.functional.conv2d(x.permute((0, 3, 1, 2)), w.permute((3, 2, 0, 1)), b, stride=1, padding=1, dilation=1, groups=16).permute((0, 2, 3, 1)),
    #     [("x", rand((1, 12, 12, 16))), ("w", rand((3, 3, 1, 16))), ("b", rand((16, )))],
    #     ", (size_t[]){1, 1}, (size_t[]){1, 1}, (size_t[]){1, 1}, 16"                                                     ),
    # ("Conv2d",      lambda x, w, b: torch.nn.functional.conv2d(x.permute((0, 3, 1, 2)), w.permute((3, 2, 0, 1)), b, stride=1, padding=1, dilation=1, groups=1).permute((0, 2, 3, 1)),
    #     [("x", rand((1, 12, 12, 16))), ("w", rand((3, 3, 16, 56))), ("b", rand((56, )))],
    #     ", (size_t[]){1, 1}, (size_t[]){1, 1}, (size_t[]){1, 1}, 1"                                                     ),
    # ("LayerNorm",   lambda x, w, b: torch.nn.functional.layer_norm(x, x.shape, w, b, eps=1e-05), 
    #     [("x", rand((6, 5))), ("w", rand((6, 5))), ("b", rand((6, 5)))  ],
    #     ", 1e-05"                                                                                                        ),

    ("abs",         lambda a: torch.abs(a),             [("a", rand16((1, 4))),                                         ]),
    ("add",         lambda a, b: a + b,                 [("a", rand16((6, 7))),       ("b", rand16((6, 7)))             ]),
    ("matmulT",     lambda a, b: a @ b.T,               [("a", rand16((6, 7))),       ("b", rand16((5, 7)))             ]),
    ("matmul",      lambda a, b: a @ b,                 [("a", rand16((6, 7))),       ("b", rand16((7, 5)))             ]),
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
  enableAcceleratorFeatures();

  size_t cycles = 0;

  {{ code }}

}
"""


def typeToStr(dtype: torch.dtype):
    if dtype == torch.float16:
        return "DTYPE_F16"
    elif dtype == torch.float32:
        return "DTYPE_F32"



def formatTensor(name: str, tensor: torch.Tensor):
    dim = len(tensor.shape)
    shape = ", ".join([str(s) for s in tensor.shape])
    dtype = typeToStr(tensor.dtype)
    data = ",".join([hex(b) for b in tensor.contiguous().numpy().flatten().tobytes()])
    human_readable = str(tensor.contiguous().numpy()).replace("\n", " ")[:80]
    tensor_str = env.from_string("""
    // {{ human_readable }}
    Tensor *{{ name }} = NN_tensor({{ dim }}, (size_t[]){ {{ shape }} }, {{ dtype }}, (uint8_t[]){ {{ data }} });""").render(
        human_readable=human_readable, name=name, dim=dim, shape=shape, dtype=dtype, data=data
    )
    return tensor_str
    


def generateTestPattern(op, function, inputs, additional_params=""):
    result = function(*[value for name, value in inputs])


    tensor_constructors = []
    tensor_destructors = []

    for name, value in inputs:
        if type(value) == torch.Tensor and name != "actual":
            dim = len(value.shape)
            shape = ", ".join([str(s) for s in value.shape])
            dtype = typeToStr(value.dtype)
            data = ", ".join([str(b) for b in value.contiguous().numpy().flatten().tobytes()])

            human_readable = str(value.contiguous().numpy()).replace("\n", " ")[:80]
            tensor_str = formatTensor(name, value)
            
            tensor_constructors.append(tensor_str)
            tensor_destructors.append("    NN_deleteTensor({});\n".format(name))

        elif type(value) == float:
            tensor_str = env.from_string("float {{ name }} = {{ value }};").render(name=name, value=value)
            tensor_constructors.append(tensor_str)
        

    golden_str = formatTensor("golden", result)

    dim = len(result.shape)
    shape = ", ".join([str(s) for s in result.shape])
    dtype = typeToStr(result.dtype)
    
    result_tensors = golden_str + env.from_string("""
    Tensor *actual = NN_zeros({{ dim }}, (size_t[]){ {{ shape }} }, {{ dtype }});""").render(
        dim=dim, shape=shape, dtype=dtype
    )

    inputs = ", ".join([name for name, value in inputs if name != "actual"])
    inputs = ", " + inputs if inputs else inputs

    func_str = env.from_string("""    NN_{{ op }}(actual{{ inputs }}{{ additional_params }});\n""").render(
        op=op, inputs=inputs, additional_params=additional_params
    )

    test_template = env.from_string("""
  {
    printf("{{ (op + ":").ljust(24) }}");
{% for tensor_str in tensor_constructors %}{{ tensor_str }}{% endfor %}

{{ result_tensors }}

    cycles = readCycles();
{{ func_str }}                          
    cycles = readCycles() - cycles;
    printf("%s  (%lu cycles)\\n", compareTensor(golden, actual, 1e-3) ? "PASS" : "FAIL", cycles);

{% for tensor_str in tensor_destructors %}{{ tensor_str }}{% endfor %}
    NN_deleteTensor(golden);
    NN_freeTensorData(actual);
    NN_deleteTensor(actual);
  }
""")
    
    return test_template.render(op=op, tensor_constructors=tensor_constructors, tensor_destructors=tensor_destructors, result_tensors=result_tensors, func_str=func_str)


template = env.from_string(c_code)



result = template.render(code="\n".join([generateTestPattern(*pattern) for pattern in test_pattern]))

with open("generated.c", "w") as f:
    f.write(result)

