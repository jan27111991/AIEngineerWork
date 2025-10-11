import pandas as pd

# Step 1: Load dataset
rawData = pd.read_csv("SalesDataset.csv")

# Step 2: Percentiles of Total Amount
print("25th percentile of Total Amount : ", rawData['Total Amount'].quantile(0.25))
print("50th percentile (Median) of Total Amount : ", rawData['Total Amount'].quantile(0.50))
print("75th percentile of Total Amount : ", rawData['Total Amount'].quantile(0.75))

# Step 3: Summary Statistics
totalAmountDescribe = rawData['Total Amount'].describe()
print("\nSummary Statistics of Total Amount:\n", totalAmountDescribe)

# Step 4: Variance
print("\nVariance in Total Amount : ", rawData['Total Amount'].var())
print("Variance in Quantity : ", rawData['Quantity'].var())

# Step 5: Correlations
print("\nCorrelation between Age and Total Amount:\n", rawData[['Age', 'Total Amount']].corr())
print("\nCorrelation between Quantity and Total Amount:\n", rawData[['Quantity', 'Total Amount']].corr())
print("\nCorrelation between Price per Unit and Total Amount:\n", rawData[['Price per Unit', 'Total Amount']].corr())

# Step 6: Overall Correlation Matrix for all selected columns
print("\nCorrelation Matrix:\n", rawData[['Age', 'Total Amount', 'Quantity', 'Price per Unit']].corr())
