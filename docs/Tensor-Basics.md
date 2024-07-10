# Tensor Basics

Tensor types are resolved dynamically, such that the API is generic, does not include multiple struct definitions, and enables multiple types within a single program. That is, there is one Tensor type. The tensor may have doubles (DTYPE_F64), float (DTYPE_F32), ints, etc. This design makes it easy to write generic code.

The underlying fundamental operators will be statically typed, and hence the tensor-level API will dynamically determine which fundamental operator to use to do the computation.


## Tensor Element in Memory

The data of the tensor must be contiguous. This is for simplifying the code framework. The side effect of this design choice is that operations like transpose will be expensive, and hence it is recommended to perform such transformations during AOT compilation process.


## Using Externally Created Data

If the data of the tensor is already allocated in memory, that memory can be viewed as a Tensor:

```c
float data[] = { 1, 2, 3,
                 4, 5, 6 };
Tensor *tensor = nn_tensor(2, (const size_t[]){2, 3}, DTYPE_F32, data);
```


## Zero-dimensional Tensors as Scalars

A scalar is represented by a Tensor object that is zero-dimensional. These Tensors hold a single value and they can be references to a single element in a larger Tensor. They can be used anywhere a Tensor is expected. 

When creating such zero-dimensional tensor, the shape will be a NULL pointer, but the size will be set to 1 and a single element worth of memory will be allocated as the data buffer.
