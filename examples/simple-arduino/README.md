# Example Working with Arduino IDE

We need to do a bit of modification to work with the Arduino environment. 

First, since it is hard to configure Arduino to load headers from other system path, we are going to copy the latest `nn.h` and other required header files to this working directory.

After generating `model.h`, we need to comment out the array definitions in there. The `INCBIN` library already provides these definitions.

Also, we need to modify the `model.bin` location to use absolute path.

The change would look somthing like this:

```diff
#ifndef __MODEL_H
#define __MODEL_H

#include "nn.h"

// load the weight data block from the model.bin file
-INCLUDE_FILE(".rodata", "./model.bin", model_weight);
+INCLUDE_FILE(".rodata", "D:/Path/to/your/model.bin", model_weight);
- extern uint8_t model_weight_data[];
- extern size_t model_weight_start[];
- extern size_t model_weight_end[];
+ // extern uint8_t model_weight_data[];
+ // extern size_t model_weight_start[];
+ // extern size_t model_weight_end[];

...
```


