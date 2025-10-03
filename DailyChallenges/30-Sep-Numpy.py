import numpy as np

def process_matrix(arr):
    row_sums = np.sum(arr, axis=1)
    col_max = np.max(arr, axis=0)
    return row_sums, col_max
#Sample Input 
input_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result_row_sums, result_col_max = process_matrix(input_array)
print("Row sums:", result_row_sums)  
print("Column max:", result_col_max)        
