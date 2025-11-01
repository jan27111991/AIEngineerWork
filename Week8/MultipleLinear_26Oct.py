import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Load Dataset 
data_dict = {
    'x1': [1, 2, 3],
    'x2': [2, 1, 4],
    'Output': [6, 8, 14] 
}
data = pd.DataFrame(data_dict)

#Define X and y
X = data[['x1', 'x2']]  # Independent variables (now two features)
y = data['Output']      # Dependent variable

#Train/Test Split
X_train = X
X_test = X
y_train = y
y_test = y

#Create and Fit the Model
model = LinearRegression()
model.fit(X_train, y_train)
# Print the coefficients to match our earlier result
print("--- Model Coefficients ---")
print(f"Intercept (β0): {model.intercept_:.1f}")
print(f"Coefficients (β1, β2): {model.coef_}")  
print("Model Equation: y = {model.intercept_:.1f} + {model.coef_[0]:.1f}*x1 + {model.coef_[1]:.1f}*x2\n")

#Make Predictions
y_pred = model.predict(X_test)
#Performance Evaluation
print("--- Model Performance Evaluation ---")
print(f"Predicted values: {y_pred}")
print(f"Actual values:    {y_test.values}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

#Graph Plotting
fig = plt.figure(figsize=(10, 7))   
ax = fig.add_subplot(111, projection='3d')
# Scatter plot of the actual data points
ax.scatter(X_test['x1'], X_test['x2'], y_test, color='blue', s=100, label='Actual Data Points')

# Create a mesh grid for x1 and x2
x1_range = np.linspace(X['x1'].min(), X['x1'].max(), 10)
x2_range = np.linspace(X['x2'].min(), X['x2'].max(), 10)
x1_range, x2_range = np.meshgrid(x1_range, x2_range)
# Predict corresponding y values using the model
y_range = model.predict(pd.DataFrame({'x1': x1_range.ravel(), 'x2': x2_range.ravel()})).reshape(x1_range.shape)

# Plot the regression plane
ax.plot_surface(x1_range, x2_range, y_range, color='orange', alpha=0.3, label='Regression Plane')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Output (y)')
ax.set_title('Multiple Linear Regression: 3D Visualization')
ax.legend()
plt.show()

# 1. Predict with new data
new_data = pd.DataFrame({'x1': [4, 5], 'x2': [5, 3]})
new_predictions = model.predict(new_data)
print("\n--- Predictions for New Data ---")
for i, row in new_data.iterrows():
    print(f"For x1 = {row['x1']}, x2 = {row['x2']} => Predicted Output (y) = {new_predictions[i]:.2f}") 
print("\n")
print(f"Model Equation: y = {model.intercept_:.1f} + {model.coef_[0]:.1f}*x1 + {model.coef_[1]:.1f}*x2\n")
