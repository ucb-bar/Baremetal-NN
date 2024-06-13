import random

import torch
import jinja2

env = jinja2.Environment()


env.globals["len"] = len
env.globals["type"] = type
env.globals["str"] = str
env.globals["torch"] = torch


c_code = """
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <rv.h>

#include "nn.h"

#ifdef RVV
  #include "riscv_vector.h"
#endif

static void enable_vector_operations() {
  #ifdef RVV
    unsigned long mstatus;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= 0x00000600 | 0x00006000 | 0x00018000;
    asm volatile("csrw mstatus, %0"::"r"(mstatus));
  #endif
}

uint8_t float_eq(float golden, float actual, float relErr) {
  return (fabs(actual - golden) < relErr) || (fabs((actual - golden) / actual) < relErr);
}

uint8_t compare(Tensor *golden, Tensor *actual) {
  for (size_t i = 0; i < golden->size; i += 1) {
    if (!float_eq(((float *)golden->data)[i], ((float *)actual->data)[i], 1e-6)) {
      return 0;
    }
  }
  return 1;
}

int main() {
  enable_vector_operations();

  size_t cycles;

  {{ code }}

}
"""

def generateImmTestPattern(op, function, inputs, detail=""):
    result = function(*[value for name, value in inputs]).item()

    test_template = env.from_string("""
  {
    printf("{{ (op + ":").ljust(24) }}");
{% for name, value in inputs %}{% if (type(value) == torch.Tensor and name != "actual") %}
    Tensor *{{ name }} = NN_tensor({{ len(value.shape) }}, (size_t[]){ {{ value.shape | join(", ") }} }, DTYPE_F32, (float[]){ {{ value.numpy().flatten() | join(", ") }} });
{% elif str(type(value)) == "<class 'float'>" %}
    float {{ name }} = {{ value }};{% endif %}{% endfor %}
    
    float golden = {{ result }};
    float actual;

    cycles = READ_CSR("mcycle");
    actual = NN_{{ op }}({{ inputs[0][0] }});
    cycles = READ_CSR("mcycle") - cycles;
    printf("%s  (%lu cycles)\\n", float_eq(golden, actual, 1e-6) ? "PASS" : "FAIL", cycles);

{% for name, value in inputs %}{% if (type(value) == torch.Tensor and name != "actual") %}
    NN_deleteTensor({{ name }});{% endif %}{% endfor %}
    
  }
""")
    
    return test_template.render(op=op, inputs=inputs, result=result, detail=detail)


def generateTestPattern(op, function, inputs, detail=""):
    result = function(*[value for name, value in inputs])

    test_template = env.from_string("""
  {
    printf("{{ (op + detail + ":").ljust(24) }}");
{% for name, value in inputs %}{% if (type(value) == torch.Tensor and name != "actual") %}
    Tensor *{{ name }} = NN_tensor({{ len(value.shape) }}, (size_t[]){ {{ value.shape | join(", ") }} }, DTYPE_F32, (float[]){ {{ value.numpy().flatten() | join(", ") }} });
{% elif str(type(value)) == "<class 'float'>" %}
    float {{ name }} = {{ value }};{% endif %}{% endfor %}
    
    Tensor *golden = NN_tensor({{ len(result.shape) }}, (size_t[]){ {{ result.shape | join(", ") }} }, DTYPE_F32, (float[]){ {{ result.numpy().flatten() | join(", ") }} });
    Tensor *actual = NN_zeros({{ len(result.shape) }}, (size_t[]){ {{ result.shape | join(", ") }} }, DTYPE_F32);

    cycles = READ_CSR("mcycle");
    NN_{{ op }}(actual{% for name, value in inputs if name != "actual" %}, {{ name }}{% endfor %});
    cycles = READ_CSR("mcycle") - cycles;
    printf("%s  (%lu cycles)\\n", compare(golden, actual) ? "PASS" : "FAIL", cycles);

{% for name, value in inputs %}{% if (type(value) == torch.Tensor and name != "actual") %}
    NN_deleteTensor({{ name }});{% endif %}{% endfor %}
    NN_deleteTensor(golden);
    NN_freeTensorData(actual);
    NN_deleteTensor(actual);
  }
""")
    
    return test_template.render(op=op, inputs=inputs, result=result, detail=detail)


template = env.from_string(c_code)

#seed
random.seed(0)
torch.manual_seed(0)


result = template.render(code="\n".join([
    generateTestPattern(
        "add",
        lambda a, b: a + b,
        (
            ("a", torch.rand((7, 7))),
            ("b", torch.rand((7, 7))),
        ),
    ),
    generateTestPattern(
        "add1",
        lambda a, b: a + b,
        (
            ("a", torch.rand((7, 7))),
            ("v", random.random()),
        ),
    ),
    generateTestPattern(
        "sub",
        lambda a, b: a - b,
        (
            ("a", torch.rand((7, 7))),
            ("b", torch.rand((7, 7))),
        ),
    ),
    generateTestPattern(
        "addInplace",
        lambda a, b: a + b,
        (
            ("actual", torch.zeros((7, 7))),
            ("b", torch.rand((7, 7))),
        )
    ),
    generateTestPattern(
        "fill",
        lambda a, v: a.fill_(v),
        (
            ("actual", torch.zeros((7, 7))),
            ("v", random.random()),
        ),
    ),
    generateTestPattern(
        "matmulT",
        lambda a, b: a @ b.T,
        (
            ("a", torch.rand((6, 7))),
            ("b", torch.rand((5, 7))),
        ),
    ),
    generateTestPattern(
        "matmul",
        lambda a, b: a @ b,
        (
            ("a", torch.rand((6, 7))),
            ("b", torch.rand((7, 5))),
        ),
    ),
    generateTestPattern(
        "maximum",
        lambda a, b: torch.maximum(a, b),
        (
            ("a", torch.rand((7, 7))),
            ("b", torch.rand((7, 7))),
        ),
    ),
    generateTestPattern(
        "minimum",
        lambda a, b: torch.minimum(a, b),
        (
            ("a", torch.rand((7, 7))),
            ("b", torch.rand((7, 7))),
        ),
    ),
    generateImmTestPattern(
        "max",
        lambda a: torch.max(a),
        (
            ("a", torch.rand((7, 7))),
        ),
    ),
    generateImmTestPattern(
        "min",
        lambda a: torch.min(a),
        (
            ("a", torch.rand((7, 7))),
        ),
    ),
    
    ]
))

with open("main.c", "w") as f:
    f.write(result)

