# Baremetal-NN

![](docs/overview.png)

Baremetal-NN is a tool for converting PyTorch models into raw C codes that can be executed standalone in a baremetal runtime on research chips. 

## Convert the model

```bash
python ./scripts/convert.py
```

the converter will dump out three files:

`nn.h`: stores the library definition.

`operators.h`: stores the operator definitions.

`weights.h`: stores the weights and biases of the network.

`model.h`: stores the code representation of the model forward pass.



### memory layout

Baremetal-NN uses the NHWC memory layout and supports up to 4-dimension tensor.

**N**: batch, **H**: height, **W**: width, **C**: channels

### Code organization

The API functions uses the following naming convention:

`NN_operator_DataType__Platform`

`operator`: the name of the operator, this should be the same as Torch and NumPy.

`DataType`: the datatype of the operands. If the datatype of the operands and results are all the same, only one datatype should be specified. Otherwise, it should be in the order of `<Operand 0>_<Operand 1>_..._<Result 0>_<Result 1>_...`

`Platform`: the platform-specific implementation. The default scalar CPU implementation omits this field.
