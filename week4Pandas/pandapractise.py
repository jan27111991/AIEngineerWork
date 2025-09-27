import pandas as pd
import numpy as np
#Scalar Series
execution_times = pd.Series([12, 15, 20, 18, 25, 30, 22])
print("Original Series:")
print(execution_times)

# Calculate middle indices
middle_start = len(execution_times) // 2 - 1
middle_end = middle_start + 3
print(f"\nMiddle three indices: {middle_start} to {middle_end-1}")
# Use loc with integer index labels
middle_three = execution_times.loc[middle_start:middle_end-1]
print(f"\nMiddle Three (using loc):")
print(middle_three)
# Create a numpy array and convert it to a panda series
array = np.array([10, 20, 23, 45, 40])
series_from_array = pd.Series(array)
print("\nSeries from Numpy Array:")
print(series_from_array)

print(series_from_array.iloc[1:4])

# Create Series from dictionary
test_cases_dict = {
    'Alex': 500,
    'Steve': 200,
    'Bob': 300
}

test_cases_series = pd.Series(test_cases_dict)
print("\nTest Cases Series:")
print(test_cases_series)
print(test_cases_series.iloc[1])
print(test_cases_series.loc['Steve':'Bob'])
