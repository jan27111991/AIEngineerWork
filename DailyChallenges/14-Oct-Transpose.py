import numpy as np

def transposeAndFlatten(arr):
    transposed_array = arr.T
    flattened_array = transposed_array.flatten()
    return (transposed_array, flattened_array)

arr = np.array([[1, 2, 3], [4, 5, 6]])
result = transposeAndFlatten(arr)
print(result)
