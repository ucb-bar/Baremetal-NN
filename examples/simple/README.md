# Simple Example

A simple example demonstrating C = A * B + D


## Generating model weights

```bash
cd ./example/simple/
python ./scripts/run.py
```

The script will generate a `model.bin` file containing the bias and weight data.


## Building

At project root, run the following commands.

```bash
# or set up towards other targets
cmake -S ./ -B ./build/ -D CMAKE_BUILD_TYPE=Debug -D CMAKE_TOOLCHAIN_FILE=./riscv-gcc.cmake
cmake --build ./build/ --target simple
```


