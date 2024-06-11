# MLP Example

An example MLP network with three fully-connected layers.

## Initial setup

```bash
mkdir ./example/mlp/build/
cd ./example/mlp/build/
cmake ..
```

## Generating model weights

```bash
cd ./example/mlp/
python ./scripts/run.py
```

The script will generate a `model.pth` file and a `model.bin` file.

## Compiling and running the program

```bash
cd ./example/mlp/build/
cmake --build . && ./mlp

spike --isa=rv64gcv_zicntr --varch=vlen:512,elen:32 fast-depth
```


