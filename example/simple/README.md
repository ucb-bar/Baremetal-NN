# Simple Example

A simple example demonstrating C = A * B + D

## Initial setup

```bash
mkdir ./example/simple/build/
cd ./example/simple/build/
cmake ..
```

## Generating model weights

```bash
cd ./example/simple/
python ./scripts/run.py
```

The script will generate a `model.pth` file and a `model.bin` file.

## Compiling and running the program

```bash
cd ./example/simple/build/
cmake --build . && ./mnist 
```


