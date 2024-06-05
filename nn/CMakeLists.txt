

set(INCLUDES
    ./
    ./add/
    ./matmul/
)

set(SOURCES
    ./add/nn_add.c
    ./matmul/nn_matmul.c
    
    ./add/nn_add_rvv.c
    ./matmul/nn_matmul_rvv.c
)

add_library(nn ${SOURCES})
target_include_directories(nn PUBLIC ${INCLUDES})

