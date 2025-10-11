#Use Standard Scalar method to transform the data [[1, 2], [3, 4], [5, 6]])
from sklearn.preprocessing import StandardScaler
import numpy as np
#Sample Data 
arr = np.array([[1,2], [3, 4], [5, 6]])
#Create StandardScaler object
scaler = StandardScaler()
#Fit and transform the data 
scaled_data = scaler.fit_transform(arr)
print("Original Data:\n", arr)
print("Scaled Data:\n", scaled_data)


