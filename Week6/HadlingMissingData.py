# sales_dataset_analysis.py

import pandas as pd

df = pd.read_csv("test_results.csv")
missingValues = df.isnull().sum()
print("Missing values in each column:\n", missingValues)

#Replace missing Duration with the mean duration
meanDuration = df['Duration'].mean()
print("\nMean Duration: ", meanDuration)
df['Duration'].fillna(meanDuration, inplace=True)
print("\nData after handling missing values:\n", df)    

#Replace missing Status with "Unknown".
df['Status'].fillna("Unknown", inplace=True)
print("\nData after handling missing values:\n", df)    

# drop any row that still contains missing values.
df.dropna(inplace=True)
print("\nData after dropping rows with missing values:\n", df)