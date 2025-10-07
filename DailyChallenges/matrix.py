import numpy as np
def matrix_operations(arr1,arr2):
    # Matrix Addition
    addition = arr1+arr2    
    # Matrix Subtraction
    subtraction = arr1-arr2
    # Matrix Multiplication
    multiplication = arr1*arr2
    
    return (addition, subtraction, multiplication)
# Sample matrices
arr1 = np.array([[1,2],[3,4]])
arr2 = np.array([[5,6],[7,8]])
output = matrix_operations(arr1,arr2)
print(output)