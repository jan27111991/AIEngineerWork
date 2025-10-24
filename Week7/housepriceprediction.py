import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the Dataset
df = pd.read_csv('house_price_regression_dataset.csv')

# Retain only Square_Footage and House_Price columns
df = df[['Square_Footage', 'House_Price']]
print("Dataset shape:", df.shape)
print("\nFirst few rows of the dataset:")
print(df.head())
# 2. Exploratory Data Analysis (EDA)
print("\nDataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Scatter plot to visualize relationship
plt.figure(figsize=(10, 6))
plt.scatter(df['Square_Footage'], df['House_Price'], alpha=0.6, color='blue')
plt.title('Relationship between Square Footage and House Price')
plt.xlabel('Square Footage')
plt.ylabel('House Price')
plt.grid(True, alpha=0.3)
plt.show()

# 3. Feature and Target Selection
X = df[['Square_Footage']]  # Independent variable
y = df['House_Price']       # Dependent variable

# 4. Train-Test Split (80-20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# 5. Model Building
model = LinearRegression()
model.fit(X_train, y_train)

# Display intercept and coefficient
print("Regression Model Results:")
print(f"Intercept (b₀): {model.intercept_:.2f}")
print(f"Coefficient (b₁): {model.coef_[0]:.2f}")

# Interpret the coefficient
print(f"\nInterpretation: For every additional square foot, the house price increases by ${model.coef_[0]:.2f}")
# 6. Prediction and Evaluation
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

# Interpretation of R² score
print(f"\nR² Interpretation: {r2*100:.1f}% of the variance in house prices can be explained by square footage.")
# 7. Visualization

# Plot 1: Regression line with actual data points
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, alpha=0.6, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Regression Line vs Actual Data Points')
plt.xlabel('Square Footage')
plt.ylabel('House Price')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted prices
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()