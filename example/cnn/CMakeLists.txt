cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME     "cnn")

set(PROJECT_INCLUDES
    inc
)

set(PROJECT_SOURCES
    main.c
)

project(${PROJECT_NAME})


add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_INCLUDES})

add_subdirectory(../../nn ./build/nn)
target_link_libraries(${PROJECT_NAME} PUBLIC nn)


