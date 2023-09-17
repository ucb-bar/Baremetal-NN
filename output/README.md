cd ./tests
make clean && make PROGRAMS=Baremetal-NN/char-rnn && spike char-rnn.riscv
