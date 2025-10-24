import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, file_path):
        #self.file_path = file_path
        self.raw_data = None
        self.cleaned_data = None
        
    def load_data(self):
        """Load the CSV file into a DataFrame"""
        try:
            self.raw_data = pd.read_csv(self.file_path)
            print(f"Data loaded successfully. Shape: {self.raw_data.shape}")
            return True
        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def remove_empty_columns(self):
        """Remove columns that are completely empty"""
        empty_cols = []
        for col in self.cleaned_data.columns:
            if self.cleaned_data[col].isna().all():
                empty_cols.append(col)
        
        if empty_cols:
            print(f"Removing empty columns: {empty_cols}")
            self.cleaned_data = self.cleaned_data.drop(columns=empty_cols)
    
    def handle_missing_values(self):
        """Fill missing values in numeric columns with median"""
        numeric_columns = ['R_Priority', 'Complexity', 'Time', 'Cost']
        
        for col in numeric_columns:
            if col in self.cleaned_data.columns:
                missing_count = self.cleaned_data[col].isna().sum()
                if missing_count > 0:
                    median_val = self.cleaned_data[col].median()
                    self.cleaned_data[col] = self.cleaned_data[col].fillna(median_val)
                    print(f"Filled {missing_count} missing values in {col} with median {median_val:.2f}")
    
    def remove_duplicates(self):
        """Remove duplicate rows from the dataset"""
        initial_count = len(self.cleaned_data)
        self.cleaned_data = self.cleaned_data.drop_duplicates()
        final_count = len(self.cleaned_data)
        removed = initial_count - final_count
        
        if removed > 0:
            print(f"Removed {removed} duplicate rows")
    
    def remove_outliers(self, column_name):
        """Remove outliers from a specific column using IQR method"""
        if column_name not in self.cleaned_data.columns:
            return 0
            
        Q1 = self.cleaned_data[column_name].quantile(0.25)
        Q3 = self.cleaned_data[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        before_count = len(self.cleaned_data)
        self.cleaned_data = self.cleaned_data[
            (self.cleaned_data[column_name] >= lower_bound) & 
            (self.cleaned_data[column_name] <= upper_bound)
        ]
        after_count = len(self.cleaned_data)
        outliers_removed = before_count - after_count
        
        if outliers_removed > 0:
            print(f"Removed {outliers_removed} outliers from {column_name}")
        
        return outliers_removed
    
    def clean_all_data(self):
        """Execute the complete data cleaning pipeline"""
        if self.raw_data is None:
            print("No data loaded. Call load_data() first.")
            return None
            
        self.cleaned_data = self.raw_data.copy()
        original_shape = self.cleaned_data.shape
        
        print("Starting data cleaning process...")
        
        self.remove_empty_columns()
        self.handle_missing_values()
        self.remove_duplicates()
        
        # Remove outliers from key columns
        outlier_columns = ['Time', 'Cost', 'R_Priority']
        for col in outlier_columns:
            self.remove_outliers(col)
        
        final_shape = self.cleaned_data.shape
        rows_removed = original_shape[0] - final_shape[0]
        cols_removed = original_shape[1] - final_shape[1]
        
        print(f"Cleaning complete. Final dataset: {final_shape}")
        print(f"Rows removed: {rows_removed}, Columns removed: {cols_removed}")
        
        return self.cleaned_data
    
    def get_cleaned_data(self):
        """Return the cleaned dataset"""
        return self.cleaned_data
    
    def get_cleaning_summary(self):
        """Print a summary of the cleaning process"""
        if self.raw_data is None or self.cleaned_data is None:
            print("No data available for summary")
            return
            
        print("\n=== CLEANING SUMMARY ===")
        print(f"Original data shape: {self.raw_data.shape}")
        print(f"Cleaned data shape: {self.cleaned_data.shape}")
        print(f"Data reduction: {((1 - self.cleaned_data.shape[0] / self.raw_data.shape[0]) * 100):.1f}%")