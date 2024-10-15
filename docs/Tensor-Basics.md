# Tensor Basics

> **Note**: Different from PyTorch, the attributes of the tensors are static, and are determined at compile time. Hence, tensor with different datatypes and dimensions are defined with separate struct definitions. That is, there is more than one Tensor type. 
>
> This way, the runtime kernel library only needs to handle the shape information of the tensors, reducing the typechecking and broadcasting overhead.


## Attributes of a Tensor

Tensor attributes describe their dimension, shape, number of elements, and datatype.

In Baremetal-NN, the dimension and datatype of the tensor are static. Tensor with different shapes and datatypes are defined with different struct definitions. 

```c
Tensor1D_F32 tensor;  // this defines a 1D tensor with float datatype

Tensor2D_F16 tensor;  // this defines a 2D tensor with half-precision floating-point (fp16) datatype
```

The maximum dimension supported is 4. These 4D tensors are used in 2D convolutions and attention layers.

The shape of the tensor is defined by an array of integers, which is a list of the lengths of each dimension. For example, a 2x3 tensor has a shape of `(size_t []){2, 3}`.

```c
Tensor2D_F32 tensor = {
  .shape = (size_t []){2, 3},
  .data = NULL,
};

printf("Shape of tensor: (%d, %d)", tensor.shape[0], tensor.shape[1]);

// alternatively, we can use the helper function to print the shape of the tensor
nn_print_shape(2, tensor.shape);
```

## Tensor Element in Memory

The data of the tensor must be contiguous. This is for simplifying the code framework. The side effect of this design choice is that operations like transpose will be expensive, and hence it is recommended to perform such transformations during AOT compilation process.


## Using Externally Created Data

If the data of the tensor is already allocated in memory, that memory can be viewed as a Tensor:

```c
float data[] = { 1, 2, 3,
                 4, 5, 6 };
Tensor tensor = {
  .shape = (size_t []){2, 3},
  .data = data,
};
```

## Zero-dimensional Tensors as Scalars

A scalar is represented by a Tensor object that is zero-dimensional. The `.data` field of Tensors hold a single value, instead of a pointer to an array of values. Additionally, they do not have a `.shape` field.

```c
Tensor0D_F32 scalar = {
  .data = 42
};
```

