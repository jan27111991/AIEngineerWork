import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load Dataset

try:
    data = pd.read_csv('Student_Performance.csv')
except FileNotFoundError:
    print("Error: 'Student_Performance.csv' file not found.")
    exit()

print("\n--- Dataset Preview ---")
print(data.head())

# Cleamse Data: Handle categorical variables if any (example shown)
if 'Extracurricular Activities' in data.columns:
    data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})


# Define Independent (X) and Dependent (y) Variables

feature_columns = [col for col in data.columns if col != 'Performance Index']
X = data[feature_columns]
y = data['Performance Index']

print("\nIndependent Variables:", feature_columns)
print("Dependent Variable: Performance Index")


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nData Split -> Training: {len(X_train)}, Testing: {len(X_test)}")


# Create and Fit the Model

model = LinearRegression()
model.fit(X_train, y_train)

# Display Coefficients

print("\n--- Model Coefficients ---")
print(f"Intercept (β₀): {model.intercept_:.2f}")
for feature, coef in zip(feature_columns, model.coef_):
    print(f"Coefficient for {feature} (β): {coef:.2f}")

# Construct model equation (only for readability)
equation = "Performance Index = " + f"{model.intercept_:.2f}"
for feature, coef in zip(feature_columns, model.coef_):
    equation += f" + ({coef:.2f} * {feature})"
print("\nModel Equation:")
print(equation)

# Make Predictions
y_pred = model.predict(X_test)

# Performance Evaluation
print("\n--- Model Performance Evaluation ---")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.2f}")
print(f"R² Score: {r2:.2f}")
# Graph Plotting: Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', edgecolors='k')
plt.xlabel('Actual Performance Index')
plt.ylabel('Predicted Performance Index')
plt.title('Actual vs Predicted Performance')
plt.grid(True)
plt.show()

# Optional 3D Plot (only if there are exactly 2 independent variables)
if len(feature_columns) == 2:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], y_test, color='blue', s=100, label='Actual Test Data')

    # Create mesh grid
    x1_range = np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 10)
    x2_range = np.linspace(X.iloc[:, 1].min(), X.iloc[:, 1].max(), 10)
    x1_range, x2_range = np.meshgrid(x1_range, x2_range)
    y_range = model.predict(pd.DataFrame({
        feature_columns[0]: x1_range.ravel(),
        feature_columns[1]: x2_range.ravel()
    })).reshape(x1_range.shape)

    ax.plot_surface(x1_range, x2_range, y_range, color='orange', alpha=0.3)
    ax.set_xlabel(feature_columns[0])
    ax.set_ylabel(feature_columns[1])
    ax.set_zlabel('Performance Index')
    ax.set_title('Multiple Linear Regression: Performance Prediction')
    ax.legend()
    plt.show()

# User Input for Prediction
print("\n--- Predict Performance Index for Custom Input ---")
try:
    user_data = {}
    for feature in feature_columns:
        val = float(input(f"Enter value for {feature}: "))
        user_data[feature] = [val]

    user_df = pd.DataFrame(user_data)
    predicted_perf = model.predict(user_df)
    print(f"\nPredicted Performance Index: {predicted_perf[0]:.2f}")

except ValueError:
    print("Invalid input. Please enter numeric values for all features.")
