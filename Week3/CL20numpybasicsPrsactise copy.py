import numpy as np

# 1. Creating
execution_times = np.array([10, 15, 20, 25, 30, 35, 40, 45])
print("Execution Times Array:", execution_times)

# 2. Indexing & Shaping
print("First element:", execution_times[0])
print("Last element:", execution_times[-1])
print("Third element:", execution_times[2])
print("Shape of array:", execution_times.shape)

# 3. Slicing
print("First 3 test times:", execution_times[0:3])
print("Every alternate test time:", execution_times[::2])

# 4. Iteration
for idx, time in enumerate(execution_times, 1):
    print(f"Test {idx} execution time: {time} seconds")

# 5. Reshaping
reshaped = execution_times.reshape(2, 4)
print("Reshaped to (2,4):\n", reshaped)

# 6. Joining
more_times = np.array([50, 55, 60, 65])
longer_array = np.concatenate([execution_times, more_times])
print("Joined array:", longer_array)

# Using axis parameter 
axis_joined = np.concatenate([execution_times,more_times], axis=0)
print("Axis joined array:", axis_joined)

# 7. Splitting
splits = np.array_split(execution_times, 5)
print("Splits:" , splits)   
for i, split in enumerate(splits, 1):
    print(f"Split {i}:", split)

split2 = np.split(execution_times, 2)
print("Split into 2 parts:", split2)