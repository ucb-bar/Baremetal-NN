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



size_t file_size;

typedef struct {
  int width;
  int height;
  unsigned char *data;
} Image;

typedef struct {
  int width;
  int height;
  uint8_t *data;
} Disp_Image;

void free_image(Image *image) {
  if (image) {
    // free(image->data);
    free(image);
  }
}

void free_disp_image(Disp_Image *image) {
  if (image) {
    // free(image->data);
    free(image);
  }
}

Image *load_image(uint8_t *file_path) {
  // Allocate memory for the image data
  unsigned char *data = file_path;
  if (!data) {
    printf("Error: Memory allocation failed\n");
    return NULL;
  }

  // Read the data from the file
  
  // Create an Image structure and populate its fields
  Image *image = (Image *)malloc(sizeof(Image));
  if (!image) {
    printf("Error: Memory allocation failed\n");
    free(data);
    return NULL;
  }

  image->height = 256;
  image->width = 256;
  // Set the image data pointer
  image->data = data;

  return image;
}


int square(char x) {
  return (int)x * (int)x;
}

// Core function computing stereoBM
Disp_Image* compute_dispartiy(Image *left, Image *right, int min_disparity, int max_disparity, int half_block_size) {
  // allocate data for disparity, use calloc for 0 initialization
  int SAD = 0;
  int min_SAD = INT32_MAX;
  int l_r, l_c, r_r, r_c;
  int height = left->height;
  int width = left->width;

  int search_range = max_disparity - min_disparity;
  int s_w = width - 2*half_block_size - search_range;
  int s_h = height - 2*half_block_size;

  int sad_iop = 0;

  signed char *disparity = (signed char *)calloc(s_w*s_h, sizeof(signed char));
  if (!disparity) {
    printf("Error: Memory allocation failed\n");
    return NULL;
  }

  // compute disparity
  // outer loop iterating over blocks
  for (int i=0+half_block_size; i<height-half_block_size; i++) {
      for (int j=0+half_block_size-min_disparity; j<width-half_block_size-max_disparity; j++) {
          // middle loop per block
          min_SAD = INT32_MAX;
          for (int offset=min_disparity; offset<max_disparity; offset++) {
              SAD = 0;
              // inner loop per pixel: compute SAD
              for (l_r = i-half_block_size; l_r < half_block_size+i; l_r++) {
                  for (l_c = j-half_block_size; l_c < half_block_size+j; l_c++) {
                      r_r = l_r;
                      r_c = l_c + offset;
                      SAD += abs(left->data[l_r*width+l_c] - right->data[r_r*width+r_c]);
                      sad_iop++;

                      // for debugging
                      // if (i == half_block_size && j == half_block_size && offset == 5){
                      //     printf("SAD: %x, l_data: %x, r_data: %x\n", SAD, left->data[l_r*width+l_c], right->data[r_r*width+r_c]);
                      // }
                  }
              }
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
      }
  }

  printf("SAD IOPs: %d\n", sad_iop);



  Disp_Image *disparity_image = (Disp_Image *)malloc(sizeof(Disp_Image));
  if (!disparity_image) {
      printf("Error: Memory allocation failed\n");
      free(disparity);
      return NULL;
  }
  disparity_image->width = s_w;
  disparity_image->height = s_h;
  disparity_image->data = disparity;
  return disparity_image;
}

int main() {

  file_size = (size_t)left_end - (size_t)left_start;

  Image *left_image = load_image(left_data);
  
  printf("Loaded left image\n");

  Image *right_image = load_image(right_data);
  
  printf("Loaded right image\n");

  Disp_Image *disparity_image = compute_dispartiy(left_image, right_image, 0, 32, 4);
  // Save the disparity image
  
  // write only the data

  printf("printing result\n");

  Tensor *img = NN_tensor(4, (const size_t[]){1, disparity_image->height, disparity_image->width, 1}, DTYPE_U8, disparity_image->data);
  Tensor *img_small = NN_zeros(4, (const size_t[]){1, disparity_image->height / 8, disparity_image->width / 4, 1}, DTYPE_U8);

  NN_interpolate(img_small, img, (float []){0.125, 0.25});

  showASCIIImage(img_small);

  
  free_image(left_image);
  free_image(right_image);
  free_disp_image(disparity_image);
  return 0;
}
