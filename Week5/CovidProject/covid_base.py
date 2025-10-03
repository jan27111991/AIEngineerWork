import pandas as pd
import numpy as np

class COVIDDataAnalyzer:
    """Base class for COVID-19 data analysis"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.load_data()
        self.clean_data()
    
    def load_data(self):
        """Load the dataset from CSV file"""
        try:
            df = pd.read_csv(self.file_path)
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print("File not found. Please check the file path.")
            return None
    
    def clean_data(self):
        """Perform basic data cleaning"""
        if self.df is not None:
            # Remove any leading/trailing whitespaces from column names
            self.df.columns = self.df.columns.str.strip()
            print("Data cleaning completed.")
    
    def get_summary(self):
        """Get basic summary of the dataset"""
        if self.df is not None:
            return self.df.info()
    
    def display_head(self, n=5):
        """Display first n rows of the dataset"""
        if self.df is not None:
            return self.df.head(n)