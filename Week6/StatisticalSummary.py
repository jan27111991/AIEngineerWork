import numpy as np
def Statistical_Summary(arr):
    mean = np.mean(arr)
    median = np.median(arr)
    std_dev = np.std(arr)

    mean_rounded = round(float(mean),2)
    median_rounded = round(float(median),2)
    std_dev_rounded = round(float(std_dev),2)

    return(mean_rounded,median_rounded,std_dev_rounded)

#Sample Input 
arr = np.array([10,20,30,40,50])
result = Statistical_Summary(arr)
print(result)