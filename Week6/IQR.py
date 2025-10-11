import pandas as pd

# Read the CSV file
df = pd.read_csv("SalesDataset_week6day2.csv")

# Convert 'Total Amount' to numeric, handling errors
df["Total Amount"] = pd.to_numeric(df["Total Amount"], errors='coerce')

# Calculate IQR
Q1 = df['Total Amount'].quantile(0.25)
Q3 = df['Total Amount'].quantile(0.75)
IQR = Q3 - Q1

print("IQR: ", IQR) 

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Lower Bound: ", lower_bound) 
print("Upper Bound: ", upper_bound)

# Identify outliers
outliers = df[(df['Total Amount'] < lower_bound) | (df['Total Amount'] > upper_bound)]
print("\nOutliers:\n", outliers)

# Save outliers to CSV
outliers.to_csv("Outliers.csv", index=False)

# Remove outliers from the dataset
df_no_outliers = df[(df['Total Amount'] >= lower_bound) & (df['Total Amount'] <= upper_bound)]
print("\nData without outliers:\n", df_no_outliers)     
# Save cleaned data to CSV
df_no_outliers.to_csv("SalesDataset_Cleaned.csv", index=False)

#reakout Session Qsn: Use Standard Scalar method to transform the data [[1, 2], [3, 4], [5, 6]])