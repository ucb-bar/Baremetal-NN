import random

import torch
import torchtune
import jinja2


class TestGenerator:

    C_CODE_TEMPLATE = """
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

    TENSOR_TEMPLATE = """
  // {{ human_readable }}
  Tensor{{ dim }}D_{{ dtype }} {{ name }} = {
    .shape = { {{ shape }} },
    .data = ({{ c_type }} *)((uint8_t[]){ {{ data }} })
  };"""

    EMPTY_TENSOR_TEMPLATE = """
  // {{ human_readable }}
  Tensor{{ dim }}D_{{ dtype }} actual = {
    .shape = { {{ shape }} },
    .data = ({{ c_type }} *)malloc(sizeof({{ c_type }}) * {{ size }})
  };"""

    TEST_BLOCK_TEMPLATE = """
  {
    printf("{{ (op + ":").ljust(24) }}");
    {% for tensor_str in tensor_constructors %}{{ tensor_str }}{% endfor %}

    {{ result_tensors }}

    cycles = read_cycles();
    {{ func_str }}                          
    cycles = read_cycles() - cycles;
    printf("%s  (%lu cycles)\\n", NN_equals{{ dim }}d_{{ dtype.lower() }}(&golden, &actual, {{ precision }}) ? "PASS" : "FAIL", cycles);

    {% for tensor_str in tensor_destructors %}{{ tensor_str }}{% endfor %}
      // NN_free_tensor_data(actual);
  }"""

    @staticmethod
    def get_type_str(tensor: torch.Tensor) -> str:
        if tensor.dtype == torch.float16:
            return "F16", "float16_t"
        elif tensor.dtype == torch.float32:
            return "F32", "float"

    def __init__(self, seed=0):

        # see if we have a GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))

        # set seed to ensure reproducibility
        self.seed = seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # create jinja2 environment
        self.env = jinja2.Environment()

        # add convenience functions to the environment
        self.env.globals["len"] = len
        self.env.globals["type"] = type
        self.env.globals["str"] = str
        self.env.globals["torch"] = torch

        self.tests = []

    def zeros(self, shape, dtype=torch.float32):
        return torch.zeros(shape, dtype=dtype, device=self.device)
    
    def ones(self, shape, dtype=torch.float32):
        return torch.ones(shape, dtype=dtype, device=self.device)

    def rand(self, shape, dtype=torch.float32):
        return (torch.rand(shape, dtype=dtype, device=self.device) - 0.5) * 10

    def zeros16(self, shape):
        return self.zeros(shape, dtype=torch.float16)

    def ones16(self, shape):
        return self.ones(shape, dtype=torch.float16)

    def rand16(self, shape):
        return self.rand(shape, dtype=torch.float16)
    

    def functional_rms_norm(self, x, w, eps):
        with torch.no_grad():
            layer = torchtune.modules.RMSNorm(dim=x.shape[0], eps=eps)
            layer.scale[:] = w
            layer.to(self.device)
            o = layer(x)
        return o


    def format_tensor(self, name: str, tensor: torch.Tensor):
        dim = len(tensor.shape)
        shape = ", ".join([str(s) for s in tensor.shape])
        dtype, c_type = self.get_type_str(tensor)
        
        data_np = tensor.detach().cpu().contiguous().numpy()
        data = ",".join([hex(b) for b in data_np.flatten().tobytes()])
        human_readable = str(data_np).replace("\n", " ")[:80]

        tensor_str = self.env.from_string(self.TENSOR_TEMPLATE).render(
            human_readable=human_readable, 
            name=name, 
            dim=dim,
            shape=shape,
            dtype=dtype,
            data=data,
            c_type=c_type
        )

        return tensor_str
        


    def add_test(self, op, function, inputs, extra_args=None):
        actual_inputs = [value for name, value in inputs if value is not None]
        result = function(*actual_inputs)

        if extra_args:
            extra_args = ", ".join([str(arg) for arg in extra_args])
            extra_args = ", " + extra_args if extra_args else extra_args
        else:
            extra_args = ""

        tensor_constructors = []
        tensor_destructors = []

        for name, value in inputs:
            if type(value) == str:
                pass
            
            if type(value) == torch.Tensor and name != "actual":
                dim = len(value.shape)
                shape = ", ".join([str(s) for s in value.shape])
                dtype, c_type = self.get_type_str(value)
                tensor_str = self.format_tensor(name, value)
                
                tensor_constructors.append(tensor_str)

            elif type(value) == float:
                tensor_str = self.env.from_string("float {{ name }} = {{ value }};").render(name=name, value=value)
                tensor_constructors.append(tensor_str)
            

        golden_str = self.format_tensor("golden", result)

        dim = len(result.shape)
        shape = ", ".join([str(s) for s in result.shape])
        dtype, c_type = self.get_type_str(result)
        size = shape.replace(", ", "*")

        actual_str = self.env.from_string(self.EMPTY_TENSOR_TEMPLATE).render(
            dim=dim, shape=shape, size=size, dtype=dtype, c_type=c_type
        )

        result_tensors = golden_str + actual_str

        inputs = ", ".join(["&"+name for name, value in inputs if name != "actual"])
        inputs = ", " + inputs if inputs else inputs

        func_str = self.env.from_string("""{{ op }}(&actual{{ inputs }}{{ extra_args }});\n""").render(
            op=op, inputs=inputs, extra_args=extra_args
        )
        
        precision = "1e-4"
        if result.dtype == torch.float16:
            precision = "1e-2"

        test_block_template = self.env.from_string(self.TEST_BLOCK_TEMPLATE)
        
        test_block = test_block_template.render(
            op=op, 
            dim=dim,
            dtype=dtype,
            tensor_constructors=tensor_constructors, 
            tensor_destructors=tensor_destructors, 
            result_tensors=result_tensors, 
            func_str=func_str, 
            precision=precision
        )

        self.tests.append(test_block)


    def generate(self, out_file: str):
        template = self.env.from_string(self.C_CODE_TEMPLATE)
        result = template.render(code="\n".join(self.tests))
        
        with open(out_file, "w") as f:
            f.write(result)




t = TestGenerator()

# ==== FP16 tests ====
t.add_test("NN_add1d_f16",   lambda a, b: a + b,                                  [("a", t.rand16((7, ))),  ("b", t.rand16((7, )))                      ])
t.add_test("NN_add2d_f16",   lambda a, b: a + b,                                  [("a", t.rand16((6, 7))), ("b", t.rand16((6, 7)))                     ])


# ==== FP32 tests ====
# add
t.add_test("NN_add1d_f32",   lambda a, b: a + b,                                  [("a", t.rand((7, ))),    ("b", t.rand((7, )))                        ])
t.add_test("NN_add2d_f32",   lambda a, b: a + b,                                  [("a", t.rand((6, 7))),   ("b", t.rand((6, 7)))                       ])

# Linear
t.add_test("NN_addmm_f32",   lambda x, w, b: torch.nn.functional.linear(x, w, b), [("x", t.rand((6, 7))),   ("w", t.rand((5, 7))), ("b", t.rand((5, ))) ])

# Relu
t.add_test("NN_relu2d_f32",  lambda x: torch.nn.functional.relu(x),               [("x", t.rand((7, 7)))                                                ])

# ELU
t.add_test("NN_elu2d_f32",   lambda x: torch.nn.functional.elu(x),                [("x", t.rand((7, 7)))],    extra_args=["1.0"])


t.generate("generated.c")


