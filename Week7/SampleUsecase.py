import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Use Case 1: Predicting Test Execution Time based on Test Case Complexity
def predict_test_execution_time():
    """
    As a Test Manager, predict how long test cycles will take based on test case complexity
    """
    # Sample data: Test Case Complexity vs Execution Time (minutes)
    data = {
        'TestComplexity': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'ExecutionTime': [5, 8, 12, 18, 25, 30, 38, 45, 52, 60, 68, 75, 83, 90, 98]
    }
    
    df = pd.DataFrame(data)
    print("Test Execution Time Prediction Dataset:")
    print(df.head())
    
    X = df[['TestComplexity']]  # Independent variable: Test complexity (1-10 scale)
    y = df['ExecutionTime']     # Dependent variable: Execution time in minutes
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'\n=== Test Execution Time Prediction Results ===')
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R² Score: {r2:.2f}')
    print(f'Coefficient: {model.coef_[0]:.2f}')
    print(f'Intercept: {model.intercept_:.2f}')
    
    # Predict for new test cases
    new_complexities = [[2], [6], [9]]
    predicted_times = model.predict(new_complexities)
    print(f'\nPredicted Execution Times:')
    for complexity, time in zip(new_complexities, predicted_times):
        print(f'Complexity {complexity[0]}: {time:.1f} minutes')
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.7)
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
    plt.xlabel('Test Case Complexity (1-10 Scale)')
    plt.ylabel('Execution Time (Minutes)')
    plt.title('Test Case Complexity vs Execution Time - QE Prediction Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model

# Use Case 2: Predicting Defect Density based on Code Complexity
def predict_defect_density():
    """
    Predict defect density based on code complexity metrics
    """
    # Sample data: Code Complexity vs Defects per KLOC
    data = {
        'CodeComplexity': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'DefectDensity': [2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30, 35, 40]
    }
    
    df = pd.DataFrame(data)
    print("\nDefect Density Prediction Dataset:")
    print(df.head())
    
    X = df[['CodeComplexity']]
    y = df['DefectDensity']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'\n=== Defect Density Prediction Results ===')
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R² Score: {r2:.2f}')
    print(f'Coefficient: {model.coef_[0]:.2f}')
    print(f'Intercept: {model.intercept_:.2f}')
    
    # Risk assessment
    complexity_threshold = 35
    predicted_defects = model.predict([[complexity_threshold]])[0]
    print(f'\nRisk Assessment:')
    print(f'At complexity {complexity_threshold}, predicted defect density: {predicted_defects:.1f} defects/KLOC')
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='red', label='Actual Defects', alpha=0.7)
    plt.plot(X_test, y_pred, color='blue', linewidth=2, label='Predicted Defects')
    plt.axhline(y=predicted_defects, color='orange', linestyle='--', label='Risk Threshold')
    plt.axvline(x=complexity_threshold, color='orange', linestyle='--')
    plt.xlabel('Code Complexity (Cyclomatic Complexity)')
    plt.ylabel('Defect Density (Defects per KLOC)')
    plt.title('Code Complexity vs Defect Density - QE Risk Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model

# Use Case 3: Predicting Test Automation ROI
def predict_automation_roi():
    """
    Predict Return on Investment for test automation based on manual test execution frequency
    """
    # Sample data: Execution Frequency vs Automation ROI
    data = {
        'ExecutionFrequency': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'ROI': [1.2, 2.5, 3.8, 5.2, 6.5, 7.8, 9.1, 10.4, 11.7, 13.0]
    }
    
    df = pd.DataFrame(data)
    print("\nTest Automation ROI Prediction Dataset:")
    print(df.head())
    
    X = df[['ExecutionFrequency']]
    y = df['ROI']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'\n=== Test Automation ROI Prediction Results ===')
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R² Score: {r2:.2f}')
    print(f'Coefficient: {model.coef_[0]:.2f}')
    print(f'Intercept: {model.intercept_:.2f}')
    
    # ROI predictions for different frequencies
    frequencies = [[8], [20], [35]]
    predicted_roi = model.predict(frequencies)
    
    print(f'\nAutomation ROI Predictions:')
    for freq, roi in zip(frequencies, predicted_roi):
        recommendation = "High Priority" if roi > 5 else "Evaluate Further"
        print(f'Frequency {freq[0]}: ROI = {roi:.1f}x - {recommendation}')
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='green', label='Actual ROI', alpha=0.7)
    plt.plot(X_test, y_pred, color='purple', linewidth=2, label='Predicted ROI')
    plt.axhline(y=5, color='red', linestyle='--', label='ROI Threshold')
    plt.xlabel('Test Execution Frequency (per release)')
    plt.ylabel('Return on Investment (ROI Multiple)')
    plt.title('Test Execution Frequency vs Automation ROI - QE Investment Planning')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model

# Main execution for Test Manager
if __name__ == "__main__":
    print("=" * 60)
    print("QUALITY ENGINEERING PREDICTIVE MODELS FOR TEST MANAGEMENT")
    print("=" * 60)
    
    # Run all QE use cases
    execution_model = predict_test_execution_time()
    defect_model = predict_defect_density()
    roi_model = predict_automation_roi()
    
    print("\n" + "=" * 60)
    print("SUMMARY FOR TEST MANAGER")
    print("=" * 60)
    print("""
    Practical Applications:
    1. Test Planning: Predict test cycle durations accurately
    2. Risk Assessment: Identify high-risk modules for focused testing
    3. Resource Allocation: Optimize test automation investments
    4. Release Decisions: Data-driven quality assessments
    
    Next Steps:
    - Collect historical project data
    - Validate models with actual project outcomes
    - Integrate predictions into test planning processes
    - Continuously refine models with new data
    """)