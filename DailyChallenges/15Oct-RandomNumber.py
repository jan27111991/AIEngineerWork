import numpy as np

def randomArrayAnalysis(n):
    arr = np.random.randint(10, 101, size=n)
    mean_val = round(np.mean(arr),2)
    max_val = np.max(arr)
    min_val = np.min(arr)
    return (int(min_val), int(max_val), int(mean_val))

n = 5
result = randomArrayAnalysis(n)
print(f"Output: {result} ")  