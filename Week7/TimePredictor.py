import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class TimePredictor:
    def __init__(self, data_frame):
        self.data = data_frame
        self.model = None
        self.training_results = {}
        self.feature_columns = ['Complexity']
    
    def prepare_data(self, test_size=0.2):
        """Prepare data for training and testing"""
        if 'Complexity' not in self.data.columns or 'Time' not in self.data.columns:
            print("Error: Required columns 'Complexity' or 'Time' not found")
            return False
        
        X = self.data[self.feature_columns]
        y = self.data['Time']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Data split: {len(self.X_train)} training, {len(self.X_test)} test samples")
        return True
    
    def train_model(self):
        """Train the linear regression model"""
        if not hasattr(self, 'X_train'):
            print("Data not prepared. Call prepare_data() first.")
            return False
        
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        
        return True
    
    def evaluate_model(self):
        """Evaluate model performance"""
        if self.model is None:
            print("Model not trained. Call train_model() first.")
            return False
        
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, self.y_pred)
        mae = np.mean(np.abs(self.y_test - self.y_pred))
        
        self.training_results = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'coefficient': self.model.coef_[0],
            'intercept': self.model.intercept_
        }
        
        print("=== MODEL EVALUATION ===")
        print(f"R² Score: {r2:.3f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"Coefficient: {self.model.coef_[0]:.2f}")
        print(f"Intercept: {self.model.intercept_:.2f}")
        
        return self.training_results
    
    def predict_for_complexity(self, complexity_values):
        """Predict time for given complexity values"""
        if self.model is None:
            print("Model not trained. Call train_model() first.")
            return None
        
        if isinstance(complexity_values, (int, float)):
            complexity_values = [complexity_values]
        
        predictions = self.model.predict(np.array(complexity_values).reshape(-1, 1))
        
        print("Prediction Results:")
        for comp, pred in zip(complexity_values, predictions):
            print(f"  Complexity {comp}: {pred:.1f} time units")
        
        return predictions
    
    def plot_regression_results(self):
        """Create visualization of regression results"""
        if self.model is None:
            print("Model not trained. Call train_model() first.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Scatter plot of actual vs predicted
        plt.scatter(self.X_test, self.y_test, alpha=0.6, color='blue', label='Actual', s=50)
        plt.scatter(self.X_test, self.y_pred, alpha=0.6, color='red', label='Predicted', s=50)
        
        # Regression line
        x_range = np.linspace(self.X_test.min().values[0], self.X_test.max().values[0], 100)
        y_range = self.model.predict(x_range.reshape(-1, 1))
        plt.plot(x_range, y_range, 'r-', linewidth=2, label='Regression Line')
        
        plt.xlabel('Test Complexity')
        plt.ylabel('Execution Time')
        plt.title('Test Complexity vs Execution Time - Prediction Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_residuals(self):
        """Plot residuals to check model assumptions"""
        if self.model is None:
            return
        
        residuals = self.y_test - self.y_pred
        
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.scatter(self.y_pred, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=15, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        """Run the complete prediction pipeline"""
        print("Starting time prediction analysis...")
        
        if not self.prepare_data():
            return None
        
        if not self.train_model():
            return None
        
        results = self.evaluate_model()
        
        # Make some sample predictions
        sample_complexities = [1, 3, 5, 7]
        self.predict_for_complexity(sample_complexities)
        
        # Create visualizations
        self.plot_regression_results()
        self.plot_residuals()
        
        return results