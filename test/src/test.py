import numpy as np



N_DIMS  = 2

def broadcast_shapes(*shapes):
    if not shapes:
        raise ValueError("No shapes to broadcast")
    
    # Determine the maximum number of dimensions
    max_len = max(len(shape) for shape in shapes)
    
    # Initialize the resulting shape with ones
    result_shape = [1] * max_len
    
    for shape in shapes:
        # Pad the shape with ones to match the maximum length
        padded_shape = [1] * (max_len - len(shape)) + list(shape)
        
        # Update the resulting shape
        for i in range(max_len):
            if result_shape[i] == 1:
                result_shape[i] = padded_shape[i]
            elif padded_shape[i] != 1 and result_shape[i] != padded_shape[i]:
                raise ValueError(f"Shapes {shapes} are not broadcastable")
    
    return tuple(result_shape)

def add_tensors_with_broadcasting(tensor1, tensor2):
    tensor1_dim = len(tensor1.shape)
    tensor2_dim = len(tensor2.shape)

    shape1 = tensor1.shape
    shape2 = tensor2.shape
    
    strides1 = tensor1.strides
    strides2 = tensor2.strides

    print("Shape 1:", shape1)
    print("Shape 2:", shape2)
    print("Strides 1:", strides1)
    print("Strides 2:", strides2)
    
    # Determine the resulting shape after broadcasting
    result_shape = broadcast_shapes(shape1, shape2)

    print("Result shape:", result_shape)

    result = np.empty(result_shape, dtype=np.float64)
    
    # Calculate the strides for broadcasting
    broadcast_strides1 = np.zeros_like(result_shape, dtype=np.intp)
    broadcast_strides2 = np.zeros_like(result_shape, dtype=np.intp)

    for i in range(N_DIMS):
        if i >= tensor1_dim or shape1[tensor1_dim-(i+1)] == 1:
            broadcast_strides1[tensor1_dim-(i+1)] = 0
        else:
            broadcast_strides1[tensor1_dim-(i+1)] = strides1[tensor1_dim-(i+1)]
        
        if i >= tensor2_dim or shape2[tensor1_dim-(i+1)] == 1:
            broadcast_strides2[tensor1_dim-(i+1)] = 0
        else:
            broadcast_strides2[tensor1_dim-(i+1)] = strides2[tensor1_dim-(i+1)]
    
    print("Broadcast strides 1:", broadcast_strides1)
    print("Broadcast strides 2:", broadcast_strides2)

    # Iterate through the result tensor and calculate the values
    it = np.nditer(result, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        idx1 = tuple(i * (s != 0) for i, s in zip(idx, broadcast_strides1))
        idx2 = tuple(i * (s != 0) for i, s in zip(idx, broadcast_strides2))

        print(idx1, idx2)
        
        result[idx] = tensor1[idx1] + tensor2[idx2]
        
        it.iternext()
    
    return result

# Example usage
tensor1 = np.array([[1, 2], [3,4], [5, 6]], dtype=np.float32)
# tensor2 = np.array([[1]])
tensor2 = np.array([[1], [2], [3]], dtype=np.float32)
result = add_tensors_with_broadcasting(tensor1, tensor2)
print(result)