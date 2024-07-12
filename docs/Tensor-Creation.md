# Tensor Creation

A set of factory functions are available for creating a tensor object. These factory functions configure the shape, data type, device and other properties of the new tensor, and optionally populate them according to specific algorithms.

## Factory Functions

A *factory function* is a function that produces a new tensor. There are many factory functions available, which differ in the way they initialize a new tensor before returning it. All factory functions adhere to the following general “schema”:

```c
Tensor *NN_<function-name>(<ndim>, <shape>, <datatype>, <tensor-options>)
```

### Available Factory Functions

The following factory functions are available at the time of this writing:

**tensor**: Returns a tensor with uninitialized values or preallocated buffer.

**zeros**: Returns a tensor filled with all zeros.

**ones**: Returns a tensor filled with all ones.

**full**: Returns a tensor filled with a single value.

**rand**: Returns a tensor filled with values drawn from a uniform distribution on [0, 1).

**randint**: Returns a tensor with integers randomly drawn from an interval.

**arange**: Returns a tensor with a sequence of integers.

