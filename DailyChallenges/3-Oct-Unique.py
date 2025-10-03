import numpy as np

def findUnique(arr):
    unique_elements = np.unique(arr)
    return unique_elements

#Sample Input 

arr = np.array([4,2,7,2,4,9])
result = findUnique(arr)
print("Unique elements in the array:", result)