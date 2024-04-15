import numpy as np


DIM = 16

A = np.linspace(0, DIM-1, DIM, dtype=np.int32)
B = np.linspace(0, DIM-1, DIM, dtype=np.int32)
D = np.linspace(0, DIM-1, DIM, dtype=np.int32)

# repeat A to 8x8
A = A.repeat(DIM).reshape(DIM, DIM)
B = B.repeat(DIM).reshape(DIM, DIM)
D = D.repeat(DIM).reshape(DIM, DIM)

# print(A @ B + D)

print(np.matmul(A, B) + D)
print((np.matmul(A, B) + D).astype(np.int8))
