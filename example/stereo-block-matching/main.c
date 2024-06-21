// This code is adapted from the following source:
// https://github.com/ankitdhall/stereo-block-matching/blob/master/sbm.cpp
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "nn.h"
#include "termimg.h"


INCLUDE_FILE(".rodata", "../data/left.bin", left);
extern uint8_t left_data[];
extern size_t left_start[];
extern size_t left_end[];

INCLUDE_FILE(".rodata", "../data/right.bin", right);
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

  signed char *disparity = (signed char *)calloc(s_w*s_h, sizeof(signed char));
  if (!disparity) {
    printf("Error: Memory allocation failed\n");
    return NULL;
  }


  Tensor *left_block = NN_tensor(2, (const size_t[]){1, 2*half_block_size}, DTYPE_U8, (uint8_t *)left->data);
  Tensor *right_block = NN_tensor(2, (const size_t[]){1, 2*half_block_size}, DTYPE_U8, (uint8_t *)right->data);
  Tensor *left_block_signed = NN_tensor(2, (const size_t[]){1, 2*half_block_size}, DTYPE_U32, NULL);
  Tensor *right_block_signed = NN_tensor(2, (const size_t[]){1, 2*half_block_size}, DTYPE_U32, NULL);
  Tensor *diff = NN_tensor(2, (const size_t[]){1, 2*half_block_size}, DTYPE_U8, NULL);
  Tensor *diff_wide = NN_tensor(2, (const size_t[]){1, 2*half_block_size}, DTYPE_I32, NULL);
  Tensor *out = NN_tensor(1, (const size_t[]){1}, DTYPE_I32, NULL);
  
  // Tensor *left_block = NN_tensor(2, (const size_t[]){1, 1}, DTYPE_U8, (uint8_t *)left->data);
  // Tensor *right_block = NN_tensor(2, (const size_t[]){1, 1}, DTYPE_U8, (uint8_t *)right->data);
  // Tensor *left_block_signed = NN_tensor(2, (const size_t[]){1, 1}, DTYPE_U32, NULL);
  // Tensor *right_block_signed = NN_tensor(2, (const size_t[]){1, 1}, DTYPE_U32, NULL);
  // Tensor *diff = NN_tensor(2, (const size_t[]){1, 1}, DTYPE_I32, NULL);
  // Tensor *out = NN_tensor(1, (const size_t[]){1}, DTYPE_I32, NULL);

  // compute disparity
  // outer loop iterating over blocks
  for (int i = half_block_size; i < height-half_block_size; i += 1) {
    printf("i: %d / %d\n", i, height-half_block_size);
    for (int j = half_block_size - min_disparity; j < width-half_block_size - max_disparity; j += 1) {
      // printf("j: %d / %d\n", j, width-half_block_size - max_disparity);
      // middle loop per block
      min_SAD = INT32_MAX;
      for (int offset = min_disparity; offset<max_disparity; offset += 1) {
        SAD = 0;


        // inner loop per pixel: compute SAD
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
        //   for (size_t col = j - half_block_size; col < half_block_size + j; col += 1) {

            left_block->data = ((uint8_t *)left->data) + row*width + col;
            right_block->data = ((uint8_t *)right->data) + row*width + col + offset;
            
            NN_sub(diff, left_block, right_block);

            // // NN_printf(diff);

            diff->dtype = DTYPE_I8;

            NN_asType(diff_wide, diff);
            diff->dtype = DTYPE_U8;

            NN_absInplace(diff_wide);

            // NN_printf(diff);

            NN_sum(out, diff_wide);
            SAD += ((int32_t *)out->data)[0];

        //   }
        }
        // printf("SAD: %d\n", SAD);
        // return NULL;


        // reduction step
        if (SAD < min_SAD) {
          // for debugging
          // if (i == half_block_size) {
          //     printf("Updated min_SAD: %x, SAD: %x, j: %d, offset: %d\n", min_SAD, SAD, j, offset);
          // }
          min_SAD = SAD;
          
          disparity[(i-half_block_size)*(s_w)+j-half_block_size] = offset;
        }
      }
      // if (j > half_block_size - min_disparity + 2)
      // return NULL;
    }
  }

  NN_freeTensorData(left_block_signed);
  NN_freeTensorData(right_block_signed);
  NN_freeTensorData(diff);
  NN_freeTensorData(out);
  NN_deleteTensor(left_block_signed);
  NN_deleteTensor(right_block_signed);
  NN_deleteTensor(diff);
  NN_deleteTensor(out);
  NN_deleteTensor(left_block);
  NN_deleteTensor(right_block);

  printf("SAD IOPs: %d\n", sad_iop);


  Tensor *disparity_image = NN_tensor(4, (const size_t[]){1, s_h, s_w, 1}, DTYPE_U8, disparity);
  return disparity_image;
}

int main() {

  file_size = (size_t)left_end - (size_t)left_start;

  Tensor *left_image = NN_tensor(4, (const size_t[]){1, IMG_HEIGHT, IMG_WIDTH, 1}, DTYPE_U8, left_data);
  Tensor *right_image = NN_tensor(4, (const size_t[]){1, IMG_HEIGHT, IMG_WIDTH, 1}, DTYPE_U8, right_data);

  Tensor *disparity_image = compute_dispartiy(left_image, right_image, 0, 32, 4);
  // Save the disparity image
  
  // write only the data

  printf("printing result\n");
  NN_printShape(disparity_image);
  printf("\n");

  Tensor *img_small = NN_zeros(4, (const size_t[]){1, disparity_image->shape[1] / 4, disparity_image->shape[2] / 2, 1}, DTYPE_U8);



  NN_interpolate(img_small, disparity_image, (float []){0.25, 0.5});

  showASCIIImage(img_small);

  return 0;
}
