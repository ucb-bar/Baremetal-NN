
import numpy as np
import torch
import cv2

input_file = "./visual_1.png"

img = cv2.imread(input_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# convert to tensor
img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
img = torch.from_numpy(img)
input_bin = img.contiguous().numpy().astype(np.float32).flatten().tobytes()

assert len(input_bin) == 1 * 3 * 224 * 224 * 4

# load array from bytes
img = np.frombuffer(input_bin, dtype=np.float32)
print(img.reshape(1, 3, 224, 224))

with open("./input.bin", "wb") as f:
    f.write(input_bin)
