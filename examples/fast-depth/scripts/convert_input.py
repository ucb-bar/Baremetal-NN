
import numpy as np
import torch
import cv2

input_file = "data/visual_1.png"

img = cv2.imread(input_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# convert to tensor
img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
img = torch.from_numpy(img)

# convert to channel-last
img = img.permute(1, 2, 0)

input_bin = img.contiguous().numpy().astype(np.float32).flatten().tobytes()

assert len(input_bin) == 1 * 224 * 224 * 3 * 4

with open("./input.bin", "wb") as f:
    f.write(input_bin)

# # load array from bytes
# img = np.frombuffer(input_bin, dtype=np.float32)
# print(img.reshape(1, 224, 224, 3))


