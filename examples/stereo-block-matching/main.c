// This code is adapted from the following source:
// https://github.com/ankitdhall/stereo-block-matching/blob/master/sbm.cpp
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "rv.h"
#include "nn.h"
#include "termimg.h"


INCLUDE_FILE(".rodata", "./data/left.bin", left);
extern uint8_t left_data[];
extern size_t left_start[];
extern size_t left_end[];

INCLUDE_FILE(".rodata", "./data/right.bin", right);
extern uint8_t right_data[];
extern size_t right_start[];
extern size_t right_end[];



#define IMG_HEIGHT    256
#define IMG_WIDTH     256



size_t file_size;

typedef struct {
  int width;
  int height;
  uint8_t *data;
} Image;

typedef struct {
  int width;
  int height;
  uint8_t *data;
} Disp_Image;



int square(char x) {
  return (int)x * (int)x;
}

// Core function computing stereoBM
Tensor* compute_dispartiy(Tensor *left, Tensor *right, int min_disparity, int max_disparity, size_t half_block_size) {
  // allocate data for disparity, use calloc for 0 initialization
  int SAD = 0;
  int min_SAD = INT32_MAX;
  int height = left->shape[1];
  int width = left->shape[2];

  int search_range = max_disparity - min_disparity;
  int s_w = width - 2 * half_block_size - search_range;
  int s_h = height - 2 * half_block_size;

  int sad_iop = 0;

  Tensor *disparity_img = NN_zeros(4, (const size_t[]){1, s_h, s_w, 1}, DTYPE_U8);
  
  Tensor *left_block = NN_tensor(2, (const size_t[]){1, 2*half_block_size}, DTYPE_U8, (uint8_t *)left->data);
  Tensor *right_block = NN_tensor(2, (const size_t[]){1, 2*half_block_size}, DTYPE_U8, (uint8_t *)right->data);
  Tensor *left_block_signed = NN_tensor(2, (const size_t[]){1, 2*half_block_size}, DTYPE_U32, NULL);
  Tensor *right_block_signed = NN_tensor(2, (const size_t[]){1, 2*half_block_size}, DTYPE_U32, NULL);
  Tensor *diff = NN_tensor(2, (const size_t[]){1, 2*half_block_size}, DTYPE_U8, NULL);
  Tensor *diff_wide = NN_tensor(2, (const size_t[]){1, 2*half_block_size}, DTYPE_I32, NULL);
  Tensor *out = NN_tensor(1, (const size_t[]){1}, DTYPE_I32, NULL);
  
  // compute disparity
  // outer loop iterating over blocks
  for (int i = half_block_size; i < height-half_block_size; i += 1) {
    printf("i: %d / %d\n", i, height-half_block_size);
    for (int j = half_block_size - min_disparity; j < width-half_block_size - max_disparity; j += 1) {
      // middle loop per block
      min_SAD = INT32_MAX;
      for (int offset = min_disparity; offset < max_disparity; offset += 1) {
        SAD = 0;


        // inner loop per pixel: compute SAD //

        // scalar version
        // for (size_t row = i - half_block_size; row < i + half_block_size; row += 1) {
        //   for (size_t col = j - half_block_size; col < j + half_block_size; col += 1) {
        //     SAD += abs((int)(((uint8_t *)left->data)[row * width + col] - ((uint8_t *)right->data)[row * width + col + offset]));
        //     // printf("%d\n", (((uint8_t *)left->data)[row * width + col] - ((uint8_t *)right->data)[row * width + col + offset]));
        //     sad_iop += 1;
        //   }
        // }

        // tensor version
        size_t row = i - half_block_size;
        size_t col = j - half_block_size;
        for (size_t row = i - half_block_size; row < half_block_size + i; row += 1) {

            left_block->data = ((uint8_t *)left->data) + row*width + col;
            right_block->data = ((uint8_t *)right->data) + row*width + col + offset;
            
            NN_sub(diff, left_block, right_block);

            diff->dtype = DTYPE_I8;
            NN_copy(diff_wide, diff);
            diff->dtype = DTYPE_U8;

            NN_abs_inplace(diff_wide);

            NN_sum(out, diff_wide);
            SAD += ((int32_t *)out->data)[0];
        }
        // reduction step
        if (SAD < min_SAD) {
          min_SAD = SAD;
          
          ((uint8_t *)disparity_img->data)[(i-half_block_size)*(s_w)+j-half_block_size] = offset;
        }
      }
    }
  }

  NN_free_tensor_data(left_block_signed);
  NN_free_tensor_data(right_block_signed);
  NN_free_tensor_data(diff);
  NN_free_tensor_data(out);
  NN_delete_tensor(left_block_signed);
  NN_delete_tensor(right_block_signed);
  NN_delete_tensor(diff);
  NN_delete_tensor(out);
  NN_delete_tensor(left_block);
  NN_delete_tensor(right_block);

  printf("SAD IOPs: %d\n", sad_iop);

  return disparity_img;
}

int main() {

  file_size = (size_t)left_end - (size_t)left_start;

  Tensor *left_image = NN_tensor(4, (const size_t[]){1, IMG_HEIGHT, IMG_WIDTH, 1}, DTYPE_U8, left_data);
  Tensor *right_image = NN_tensor(4, (const size_t[]){1, IMG_HEIGHT, IMG_WIDTH, 1}, DTYPE_U8, right_data);

  size_t cycles = READ_CSR("cycle");
  Tensor *disparity_img = compute_dispartiy(left_image, right_image, 0, 32, 4);
  cycles = READ_CSR("cycle") - cycles;

  printf("Cycles: %lu\n", cycles);

  // Save the disparity image

  printf("Result:\n");
  NN_print_shape(disparity_img);
  printf("\n");

  Tensor *disparity_img_small = NN_zeros(4, (const size_t[]){1, disparity_img->shape[1] / 4, disparity_img->shape[2] / 2, 1}, DTYPE_U8);
  NN_interpolate(disparity_img_small, disparity_img, (float []){0.25, 0.5});

  show_ASCII_image(disparity_img_small, 0, 32);

  return 0;
}
