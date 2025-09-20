import numpy as np
#Create a 1D array
arr1 = np.array([10, 20, 30, 40, 50,60 ])
print("1D Array:", arr1)

# Slice the array using [2:]
print("Slice [2:]:", arr1[2:])

# Slice the array using [3:5]
print("Slice [3:5]:", arr1[3:5])

# Access the element using [-4]
print("Element at [-4]:", arr1[-4])

# Reverse the array
print("Reversed array:", arr1[::-1])

#Dimension of 1D array
print("Dimension of 1D array:", arr1.ndim)
print("==>" * 20)

# Create a 2D array
samplearray = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print("\n2D Array:\n", samplearray)

# Access specific elements
print("samplearray[0][1]:", samplearray[0][1])
print("samplearray[1][1]:", samplearray[1][1])
print("samplearray[0][3]:", samplearray[0][3])

# Calculate the sum of all elements in the 2D array
print("Sum of all elements in 2D array:", np.sum(samplearray))
sum = 0
for row in samplearray:
    print("Row:", row)
    for col in row:
        sum = sum + col
print("Sum using loop:", sum)


#Dimension of 2D array
print("Dimension of 2D array:", samplearray.ndim)
print("==>" * 20)
# Create a 3D array
sample3Dim = np.array([[[1, 2,3], [3, 4,6]], [[15, 16, 17], [18, 19,20]]])

print("sample3Dim[0][0][0]:", sample3Dim[0][0][0])
print("sample3Dim[0][1][2]:,", sample3Dim[0][1][2])
print("sample3Dim[1][1][2]:", sample3Dim[1][1][2])

#Dimension of 3D array
print("Dimension of 3D array:", sample3Dim.ndim)