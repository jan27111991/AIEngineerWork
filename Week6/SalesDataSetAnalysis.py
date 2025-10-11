# sales_dataset_analysis.py

import pandas as pd

# -------------------------
# Step 1: Load the dataset
# -------------------------
# Make sure SalesDataset.csv is in the same folder as this script
df = pd.read_csv("SalesDataset.csv")

# -------------------------
# Step 2: Perform Analysis
# -------------------------

# 1. 25th percentile of Total Amount
percentile_25 = df['Total Amount'].quantile(0.25)

# 2. 50th percentile (Median) of Total Amount
median_50 = df['Total Amount'].quantile(0.50)

# 3. 75th percentile of Total Amount
percentile_75 = df['Total Amount'].quantile(0.75)

# 4. Variance in Total Amount
variance_total_amount = df['Total Amount'].var()

# 5. Variance in Quantity sold
variance_quantity = df['Quantity'].var()
variance_quantity = df['Quantity'].var()

# 6. Correlation between Age and Total Amount
correlation_age_amount = df['Age'].corr(df['Total Amount'])

# 7. Correlation between Quantity and Total Amount
correlation_quantity_amount = df['Quantity'].corr(df['Total Amount'])

# 8. Correlation between Price per Unit and Total Amount
correlation_price_amount = df['Price per Unit'].corr(df['Total Amount'])

# -------------------------
# Step 3: Display Results
# -------------------------
print("---- Sales Data Analysis ----")
print(f"1. 25th Percentile of Total Amount: {percentile_25:.2f}")
print(f"2. Median (50th Percentile) of Total Amount: {median_50:.2f}")
print(f"3. 75th Percentile of Total Amount: {percentile_75:.2f}")
print(f"4. Variance in Total Amount: {variance_total_amount:.2f}")
print(f"5. Variance in Quantity sold: {variance_quantity:.2f}")
print(f"6. Correlation between Age and Total Amount: {correlation_age_amount:.2f}")
print(f"7. Correlation between Quantity and Total Amount: {correlation_quantity_amount:.2f}")
print(f"8. Correlation between Price per Unit and Total Amount: {correlation_price_amount:.2f}")
