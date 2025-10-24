##Use Robust Scalar method to transform the data [[1 2][3 4][5 6] [7 8] [9 10]]
from sklearn.preprocessing import RobustScaler
import numpy as np
#Sample Data
arr = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])
#Create Robust Scalar object
scaler = RobustScaler()
#Fit and transform the data
scaled_data = scaler.fit_transform(arr)
print("Original Data:\n", arr)
print("Scaled Data:\n", scaled_data)    
