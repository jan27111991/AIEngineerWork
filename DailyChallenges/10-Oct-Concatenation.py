import numpy as np

def concatenate_arrays(arr1,arr2):
    if(arr1.shape[0] != arr2.shape[0]) or arr1.shape[1] != arr2.shape[1]:
        raise "Concatenation not possible"
    
    horizontal_concat = np.concatenate((arr1, arr2), axis=1)
    vertical_concat = np.concatenate((arr1, arr2), axis=0)  
    return horizontal_concat, vertical_concat
#Sample Input 
arr1 = np.array([[1, 2],[3, 4]])
arr2 = np.array([[5, 6],[7, 8]])
result = concatenate_arrays(arr1, arr2)
print("Horizontal Concatenation:\n", result[0]) 
print("Vertical Concatenation:\n", result[1])
