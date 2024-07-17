# Baremetal-NN

![](docs/overview.png)

Baremetal-NN is a tool for converting PyTorch models into raw C codes that can be executed standalone in a baremetal runtime on research chips. 

> Note:
> After a discussion with [@iansseijelly](https://github.com/iansseijelly), we decided to switch to the simpler way of assuming array will be contiguous, and therefore directly use shape to index into elements, instead of the more generic strided access. The previous strided implementation can be access on the ["strided"](https://github.com/ucb-bar/Baremetal-NN/tree/strided) branch.


## Run Test

### Building for x86

first, we clean any previous builds

```bash
rm -rf ./build/
```

```bash
cmake . -S ./ -B ./build/ -D CMAKE_BUILD_TYPE=Debug
cmake --build ./build/ --target tests
./build/tests/tests
```

### Building for RISC-V

first, we clean any previous builds

```bash
rm -rf ./build/
```

```bash
# make sure $RISCV is set
cmake . -D CMAKE_TOOLCHAIN_FILE=./riscv-gcc.cmake -S ./ -B ./build/ -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Debug
cmake --build ./build/ --target all
spike ./build/tests/tests 
```

### Building for RISC-V with Vector Support

first, we clean any previous builds

```bash
rm -rf ./build/
```

```bash
# make sure $RISCV is set
cmake . -D CMAKE_TOOLCHAIN_FILE=./riscv-gcc.cmake -S ./ -B ./build/ -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Debug -D RVV=ON
cmake --build ./build/ --target all
spike --isa=rv64gcv_zicntr_zfh --varch=vlen:512,elen:32 ./build/tests/tests
```

Running with FP16 support

```bash
spike --isa=rv64gcv_zicntr_zfh_zvfh --varch=vlen:512,elen:32 ./build/tests/tests
```

### Building for RISC-V with Gemmini

first, we clean any previous builds

```bash
rm -rf ./build/
```

```bash
cmake . -D CMAKE_TOOLCHAIN_FILE=./riscv-gcc.cmake -S ./ -B ./build/ -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Debug -D GEMMINI=ON
cmake --build ./build/ --target all
spike --extension=gemmini ./example
```

### Cleaning build files

```
cmake --build ./build/ --target clean
```

### Cleaning CMake files

```
rm -rf ./build/
```


## Convert the model

```bash
python ./scripts/convert.py
```

the converter will dump out three files:

`nn.h`: stores the library definition.

`operators.h`: stores the operator definitions.

`weights.h`: stores the weights and biases of the network.

`model.h`: stores the code representation of the model forward pass.



### Memory layout

Baremetal-NN uses the NHWC memory layout and supports up to 4-dimension tensor.

**N**: batch, **H**: height, **W**: width, **C**: channels

### Code organization

The torch-like functions that operates on Tensor datatypes are under `nn/functional`.

The low-level implementations of kernels are under `nn/impl/<device>`.

For the low-level functions, the following naming convention is used:

`void NN__operator_datatype(size_t n, <datatype *output_ptr, size_t increment>, <datatype *input_ptr, size_t increment>);`

`operator`: the name of the operator, such as `add`, `max`.

`dataType`: the datatype of the operands, such as `i8`, `u16`, `f32`. If the datatype of the results and operands are different, it will be named `<operand 0>_<operand 1>_..._to_<result 0>_...`

`output_ptr` / `input_ptr`: the pointer to the data buffer with the correct type. The correct pointer type saves the repetitive casting within the function source code.

`increment`: the number of element to increment in order to access the next element in the buffer, **in number of elements**, not bytes. (e.g. for `f32` type, increment of 1 will access next element starting from the next 4th byte, and hence the next contiguous fp32 number.)


