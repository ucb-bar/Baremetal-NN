# FastDepth

Example project of converting the [ICRA 2019 "FastDepth: Fast Monocular Depth Estimation on Embedded Systems"](https://github.com/dwofk/fast-depth) to baremetal environment.

## Initial setup

```bash
mkdir ./example/fast-depth/build/
cd ./example/fast-depth/build/
cmake .. -DRISCV=ON
```

## Generating model weights

TODO

The script will generate a `model.bin` file.


## Generating model inputs

```bash
python process_input.py
```

The script will generate a `input.bin` file.


## Compiling and running the program

Building for x86 CPU:
```bash
cd ./example/fast-depth/build/
cmake .. -D X86=ON
cmake --build .
./fast-depth
```

Building for spike:

```bash
cd ./example/fast-depth/build/
cmake .. -D RISCV=ON
cmake --build .
spike --isa=rv64gcv_zicntr --varch=vlen:512,elen:32 ./fast-depth
```


