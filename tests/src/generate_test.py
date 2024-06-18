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


def rand(shape, dtype: torch.dtype = torch.float32):
    if dtype == torch.float32:
        return (torch.rand(shape, dtype=dtype) - 0.5) * 100
    elif dtype == torch.float16:
        return (torch.rand(shape, dtype=dtype) - 0.5) * 10


test_pattern = [
    # ("abs",         lambda a: torch.abs(a),             [("a", rand((7, 7))),                                           ]),
    # ("add",         lambda a, b: a + b,                 [("a", rand((7, 7))),         ("b", rand((7, 7)))               ]),
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
    
    # # ("Linear",      lambda x, w, b: torch.nn.functional.linear(x, w, b), [("x", rand((6, 7))), ("w", rand((5, 7))), ("b", rand((1, 5)))]),
    # ("Linear",      lambda x, w, b: torch.nn.functional.linear(x, w, b), [("x", rand((6, 7))), ("w", rand((5, 7))), ("b", rand((1, 5)))]),
    # ("ReLU",        lambda x: torch.nn.functional.relu(x),      [("x", rand((7, 7)))                                    ]),
    # ("ReLU6",       lambda x: torch.nn.functional.relu6(x),     [("x", rand((7, 7)))                                    ]),
    # ("Conv2d",      lambda x, w, b: torch.nn.functional.conv2d(x.permute((0, 3, 1, 2)), w.permute((3, 2, 0, 1)), b, stride=1, padding=0, dilation=1, groups=1).permute((0, 2, 3, 1)),
    #     [("x", rand((1, 16, 16, 3))), ("w", rand((3, 3, 3, 6))), ("b", rand((6, )))],
    #     ", (size_t[]){1, 1}, (size_t[]){0, 0}, (size_t[]){1, 1}, 1"
    # ),
    # ("Conv2d",      lambda x, w, b: torch.nn.functional.conv2d(x.permute((0, 3, 1, 2)), w.permute((3, 2, 0, 1)), b, stride=1, padding=1, dilation=1, groups=1).permute((0, 2, 3, 1)),
    #     [("x", rand((1, 16, 16, 3))), ("w", rand((3, 3, 3, 6))), ("b", rand((6, )))],
    #     ", (size_t[]){1, 1}, (size_t[]){1, 1}, (size_t[]){1, 1}, 1"
    # ),

    ("abs",         lambda a: torch.abs(a),             [("a", rand((1, 4), dtype=torch.float16)),                      ]),
]


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


def generateTestPattern(op, function, inputs, additional_params=""):
    result = function(*[value for name, value in inputs])

    input_bytes = []

    for name, value in inputs:
        if type(value) == torch.Tensor:
            value_bytes = ", ".join([hex(b) for b in value.contiguous().numpy().flatten().tobytes()])
            input_bytes.append(value_bytes)
    
    output_bytes = ", ".join([hex(b) for b in result.contiguous().numpy().flatten().tobytes()])

    if result.dtype == torch.float32:
        dtype_str = "DTYPE_F32"
    elif result.dtype == torch.float16:
        dtype_str = "DTYPE_F16"

    test_template = env.from_string("""
  {
    printf("{{ (op + ":").ljust(24) }}");
{% for name, value in inputs %}{% if (type(value) == torch.Tensor and name != "actual") %}
    // {{ value.contiguous().numpy().flatten() | join(", ") }}
    Tensor *{{ name }} = NN_tensor({{ len(value.shape) }}, (size_t[]){ {{ value.shape | join(", ") }} }, {{ dtype_str }}, (uint8_t[]){ {{ input_bytes.pop() }} });
{% elif str(type(value)) == "<class 'float'>" %}
    float {{ name }} = {{ value }};{% endif %}{% endfor %}

    // {{ result.contiguous().numpy().flatten() | join(", ") }}
    Tensor *golden = NN_tensor({{ len(result.shape) }}, (size_t[]){ {{ result.shape | join(", ") }} }, {{ dtype_str }}, (uint8_t[]){ {{ output_bytes }} });
    Tensor *actual = NN_zeros({{ len(result.shape) }}, (size_t[]){ {{ result.shape | join(", ") }} }, {{ dtype_str }});
    cycles = readCycles();
    NN_{{ op }}(actual{% for name, value in inputs if name != "actual" %}, {{ name }}{% endfor %}{{ additional_params }});
    cycles = readCycles() - cycles;
    printf("%s  (%lu cycles)\\n", compareTensor(golden, actual, 1e-3) ? "PASS" : "FAIL", cycles);

{% for name, value in inputs %}{% if (type(value) == torch.Tensor and name != "actual") %}
    NN_deleteTensor({{ name }});{% endif %}{% endfor %}
    NN_deleteTensor(golden);
    NN_freeTensorData(actual);
    NN_deleteTensor(actual);
  }
""")
    
    return test_template.render(op=op, inputs=inputs, result=result, additional_params=additional_params, dtype_str=dtype_str, input_bytes=input_bytes, output_bytes=output_bytes)


template = env.from_string(c_code)



result = template.render(code="\n".join([generateTestPattern(*pattern) for pattern in test_pattern]))

with open("generated.c", "w") as f:
    f.write(result)

