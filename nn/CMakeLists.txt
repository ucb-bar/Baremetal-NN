

set(INCLUDES
    inc
)

set(SOURCES
    src/nn.c
    src/nn_add.c
    src/nn_elu.c
    src/nn_relu.c
    src/nn_clip.c
    src/nn_linear.c
    src/nn_matmul.c
    src/nn_transpose.c
)

add_library(nn ${SOURCES})
target_include_directories(nn PUBLIC ${INCLUDES})

