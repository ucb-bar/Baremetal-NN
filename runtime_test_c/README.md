
```bash
cd ./runtime_test_c
make clean
make PROGRAMS=char-rnn
spike char-rnn.riscv

make clean && make PROGRAMS=char-rnn && spike char-rnn.riscv
```
