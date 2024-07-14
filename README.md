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



### memory layout

Baremetal-NN uses the NHWC memory layout and supports up to 4-dimension tensor.

**N**: batch, **H**: height, **W**: width, **C**: channels

### Code organization

The API functions uses the following naming convention:

`NN_operator_DataType__Platform`

`operator`: the name of the operator, this should be the same as Torch and NumPy.

`DataType`: the datatype of the operands. If the datatype of the operands and results are all the same, only one datatype should be specified. Otherwise, it should be in the order of `<Operand 0>_<Operand 1>_..._<Result 0>_<Result 1>_...`

`Platform`: the platform-specific implementation. The default scalar CPU implementation omits this field.
