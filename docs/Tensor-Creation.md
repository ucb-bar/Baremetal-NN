# Tensor Creation

Tensors can be initialized in various ways. A set of factory functions are available for creating a tensor object. These factory functions configure the shape, data type, device and other properties of the new tensor, and optionally populate them according to specific algorithms.

## Factory Functions

A *factory function* is a function that produces a new tensor. There are many factory functions available, which differ in the way they initialize a new tensor before returning it. All factory functions adhere to the following general "schema”:

```c
Tensor *nn_{creation-function}_{ndim}d_{datatype}({shape}, {options});
```

### Available Factory Functions

The following factory functions are available at the time of this writing:

#### nn_tensor()

Returns a tensor with uninitialized values or preallocated buffer.

When passing NULL as the data buffer, the method will allocate a new chunk of uninitialized data chunk.

```c
Tensor *tensor = nn_tensor2d_f32((size_t []){ 2, 2 }, NULL);
```

Alternatively, tensor be created directly from an existing data buffer.

```c
// data = [[1, 2], [3, 4]]
float data[] = { 1, 2, 3, 4 };
Tensor *tensor = nn_tensor2d_f32((size_t []){ 2, 2 }, data);
```

#### nn_zeros()

Returns a tensor filled with all zeros.

#### nn_ones()

Returns a tensor filled with all ones.

#### nn_full()

Returns a tensor filled with a single value.

#### nn_rand()

Returns a tensor filled with values drawn from a uniform distribution on [0, 1).
