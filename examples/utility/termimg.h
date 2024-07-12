#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#include "nn_tensor.h"


static const size_t NUM_MAPPING = 92;

static const uint8_t ASCII_MAP[] = {0X20, 0X60, 0X2E, 0X2D, 0X27, 0X3A, 0X5F, 0X2C, 0X5E, 0X3D, 0X3B, 0X3E, 0X3C, 0X2B, 0X21, 0X72, 0X63, 0X2A, 0X2F, 0X7A, 0X3F, 0X73, 0X4C, 0X54, 0X76, 0X29, 0X4A, 0X37, 0X28, 0X7C, 0X46, 0X69, 0X7B, 0X43, 0X7D, 0X66, 0X49, 0X33, 0X31, 0X74, 0X6C, 0X75, 0X5B, 0X6E, 0X65, 0X6F, 0X5A, 0X35, 0X59, 0X78, 0X6A, 0X79, 0X61, 0X5D, 0X32, 0X45, 0X53, 0X77, 0X71, 0X6B, 0X50, 0X36, 0X68, 0X39, 0X64, 0X34, 0X56, 0X70, 0X4F, 0X47, 0X62, 0X55, 0X41, 0X4B, 0X58, 0X48, 0X6D, 0X38, 0X52, 0X44, 0X23, 0X24, 0X42, 0X67, 0X30, 0X4D, 0X4E, 0X57, 0X51, 0X25, 0X26, 0X40};

static const float BRIGHTNESS_MAP[] = {0, 0.0751, 0.0829, 0.0848, 0.1227, 0.1403, 0.1559, 0.185, 0.2183, 0.2417, 0.2571, 0.2852, 0.2902, 0.2919, 0.3099, 0.3192, 0.3232, 0.3294, 0.3384, 0.3609, 0.3619, 0.3667, 0.3737, 0.3747, 0.3838, 0.3921, 0.396, 0.3984, 0.3993, 0.4075, 0.4091, 0.4101, 0.42, 0.423, 0.4247, 0.4274, 0.4293, 0.4328, 0.4382, 0.4385, 0.442, 0.4473, 0.4477, 0.4503, 0.4562, 0.458, 0.461, 0.4638, 0.4667, 0.4686, 0.4693, 0.4703, 0.4833, 0.4881, 0.4944, 0.4953, 0.4992, 0.5509, 0.5567, 0.5569, 0.5591, 0.5602, 0.5602, 0.565, 0.5776, 0.5777, 0.5818, 0.587, 0.5972, 0.5999, 0.6043, 0.6049, 0.6093, 0.6099, 0.6465, 0.6561, 0.6595, 0.6631, 0.6714, 0.6759, 0.6809, 0.6816, 0.6925, 0.7039, 0.7086, 0.7235, 0.7302, 0.7332, 0.7602, 0.7834, 0.8037, 0.9999};

/**
 * Display the image in ASCII format
 * 
 * @param tensor: input tensor
 */
void show_ASCII_image(Tensor *tensor, float min, float max) {
  assert(tensor->ndim == 4);
  assert(tensor->shape[0] == 1);
  assert(tensor->shape[3] == 1);
  
  size_t width;
  size_t height;

  if (tensor->ndim == 4) {
    height = tensor->shape[1];
    width = tensor->shape[2];
  } else if (tensor->ndim == 3) {
    height = tensor->shape[1];
    width = tensor->shape[2];
  } else if (tensor->ndim == 2) {
    height = tensor->shape[0];
    width = tensor->shape[1];
  } else {
    printf("[ERROR] Invalid tensor shape\n");
  }

  // find the min and max pixel value
  if (max == 0 && min == 0) {
    min = 1000;
    max = -1000;

    for (size_t h = 0; h < height; h += 1) {
      for (size_t w = 0; w < width; w += 1) {
        float pixel_value = ((float *)tensor->data)[h * width + w];
        
        if (pixel_value < min) {
          min = pixel_value;
        }
        if (pixel_value > max) {
          max = pixel_value;
        }
      }
    }
  }

  for (size_t h = 0; h < height; h += 1) {
    for (size_t w = 0; w < width; w += 1) {
      float pixel_value = 0;
      if (tensor->dtype == DTYPE_F32) {
        pixel_value = ((float *)tensor->data)[h * width + w];
      }
      else if (tensor->dtype == DTYPE_U8) {
        pixel_value = (float)((uint8_t *)tensor->data)[h * width + w];
      }

      // normalize the pixel value to the range [0, 1]
      pixel_value = (pixel_value - min) / (max - min);

      // find the closest brightness value in the brightness_map
      size_t brightness_index = 0;
      for (size_t i = 0; i < NUM_MAPPING; i += 1) {
        if (pixel_value < BRIGHTNESS_MAP[i]) {
          break;
        }
        brightness_index = i;
      }

      // print the corresponding ASCII character
      printf("%c", ASCII_MAP[brightness_index]);
    }
    printf("\n");
  }
}
