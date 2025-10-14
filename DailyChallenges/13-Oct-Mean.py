import numpy as np

def filtergreaterthanmean(arr):
    mean_value = np.mean(arr)
    filtered_array = arr[arr > mean_value]
    return filtered_array

# Sample Input
input_array = np.array([10,20,30,40,50])
result = filtergreaterthanmean(input_array)
print("Filtered array:", result)         