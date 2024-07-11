make && spike --isa=rv64gcv_zicntr --varch=vlen:512,elen:32 --misaligned ./llama2


```bash
wget -P checkpoints/ https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```




# Performance Benchmark

Native impl (150479)

```bash
[100%] Built target llama2
Llama 2: a small transformer model for text generation
forward taking 458394741 cycles
Once
forward taking 458545220 cycles
 upon
forward taking 458648904 cycles
 a
forward taking 458744673 cycles
 time
forward taking 458850691 cycles
,
forward taking 458942021 cycles
 there
forward taking 459044968 cycles
 was
```

Replace matmul and softmax with JIT tensors

```bash
[100%] Built target llama2
Llama 2: a small transformer model for text generation
forward taking 11942828 cycles
Once
forward taking 12093307 cycles
 upon
forward taking 12196991 cycles
 a
forward taking 12292760 cycles
 time
forward taking 12398778 cycles
,
forward taking 12490108 cycles
 there
forward taking 12593055 cycles
 was
```



