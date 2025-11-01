import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split  

#Load Dataset 
try:
    data = pd.read_csv('salary.csv')
except FileNotFoundError:
    print("Error: 'salary.csv' file not found.")
    exit()

#Define X and y
X = data[['YearsExperience', 'Rating']]  # Independent variables
y = data['Salary']                     # Dependent variable

#Train/Test Split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

#Create and Fit the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the coefficients
print("--- Model Coefficients ---")
print(f"Intercept (β0): {model.intercept_:.2f}")
print(f"Coefficients (β1, β2): {model.coef_}")
print(f"Model Equation: Salary = {model.intercept_:.2f} + {model.coef_[0]:.2f}*YearsExperience + {model.coef_[1]:.2f}*Rating\n")

#Make Prediction
y_pred = model.predict(X_test)

#Performance Evaluation
print("--- Model Performance Evaluation ---")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

#Graph Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# Scatter plot of the actual data points
ax.scatter(X_test['YearsExperience'], X_test['Rating'], y_test, color='blue', s=100, label='Actual Test Data')
# Create a mesh grid for YearsExperience and Rating
x1_range = np.linspace(X['YearsExperience'].min(), X['YearsExperience'].max(), 10)
x2_range = np.linspace(X['Rating'].min(), X['Rating'].max(), 10)
x1_range, x2_range = np.meshgrid(x1_range, x2_range)

# Predict corresponding Salary values using the model
y_range = model.predict(pd.DataFrame({'YearsExperience': x1_range.ravel(), 'Rating': x2_range.ravel()})).reshape(x1_range.shape)
# Plot the regression plane
ax.plot_surface(x1_range, x2_range, y_range, color='orange', alpha=0.3)
ax.set_xlabel('Years of Experience')
ax.set_ylabel('Rating')
ax.set_zlabel('Salary')
ax.set_title('Multiple Linear Regression: Salary Prediction')
ax.legend()
plt.show()

#Get user input for prediction
try:
    years_experience = float(input("Enter Years of Experience: "))
    if years_experience < 0:
        raise ValueError("Years of Experience cannot be negative.")
    rating = float(input("Enter Rating: "))
    if rating < 0:
        raise ValueError("Rating cannot be negative.")
    user_input = pd.DataFrame({'YearsExperience': [years_experience], 'Rating': [rating]})
    predicted_salary = model.predict(user_input)
    print(f"Predicted Salary for {years_experience} years of experience and rating {rating} is: ${predicted_salary[0]:.2f}")
except ValueError:
    print("Invalid input. Please enter numeric values for Years of Experience and Rating.")
    





