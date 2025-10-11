import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class CovidEDA:
    def __init__(self, file_path):
        """Initialize the CovidEDA class with the dataset file path"""
        self.file_path = file_path
        self.df = None
        self.df_cleaned = None
        self.df_normalized = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the dataset and keep only Confirmed and New cases columns"""
        try:
            # Load the dataset
            self.df = pd.read_csv(self.file_path)
            
            # Keep only the required columns
            required_columns = ['Confirmed', 'New cases']
            
            # Check if required columns exist
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                raise KeyError(f"Missing columns: {missing_columns}")
                
            self.df = self.df[required_columns].copy()
            print("Dataset loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            print("\nFirst 5 rows of the dataset:")
            print(self.df.head())
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def compute_statistics(self):
        """Compute and display statistical measures"""
        if self.df is None:
            print("Please load the data first!")
            return
        
        print("\n" + "="*50)
        print("STATISTICAL MEASURES")
        print("="*50)
        
        # Calculate statistics for each column
        for column in self.df.columns:
            print(f"\n--- {column} ---")
            print(f"Mean: {self.df[column].mean():.2f}")
            print(f"Median: {self.df[column].median():.2f}")
            print(f"Variance: {self.df[column].var():.2f}")
            print(f"Standard Deviation: {self.df[column].std():.2f}")
            print(f"Min: {self.df[column].min():.2f}")
            print(f"Max: {self.df[column].max():.2f}")
        
        # Correlation matrix
        print(f"\n--- Correlation Matrix ---")
        correlation_matrix = self.df.corr()
        print(correlation_matrix)
        
        return correlation_matrix
    
    def detect_outliers_iqr(self):
        """Detect and remove outliers using IQR method"""
        if self.df is None:
            print("Please load the data first!")
            return
        
        print("\n" + "="*50)
        print("OUTLIER DETECTION USING IQR")
        print("="*50)
        
        self.df_cleaned = self.df.copy()
        
        for column in self.df.columns:
            # Calculate Q1, Q3, and IQR
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
            
            print(f"\n--- {column} ---")
            print(f"Q1 (25th percentile): {Q1:.2f}")
            print(f"Q3 (75th percentile): {Q3:.2f}")
            print(f"IQR: {IQR:.2f}")
            print(f"Lower bound: {lower_bound:.2f}")
            print(f"Upper bound: {upper_bound:.2f}")
            print(f"Number of outliers: {len(outliers)}")
            print(f"Outlier values: {outliers[column].values}")
            
            # Remove outliers
            self.df_cleaned = self.df_cleaned[
                (self.df_cleaned[column] >= lower_bound) & 
                (self.df_cleaned[column] <= upper_bound)
            ]
        
        print(f"\nOriginal dataset shape: {self.df.shape}")
        print(f"Cleaned dataset shape (after outlier removal): {self.df_cleaned.shape}")
        print(f"Rows removed: {len(self.df) - len(self.df_cleaned)}")
        
        print("\nFirst 5 rows of cleaned dataset:")
        print(self.df_cleaned.head())
        
        return self.df_cleaned
    
    def normalize_data(self):
        """Normalize data using StandardScaler"""
        if self.df_cleaned is None:
            print("Please detect and remove outliers first!")
            return
        
        print("\n" + "="*50)
        print("DATA NORMALIZATION USING STANDARD SCALER")
        print("="*50)
        
        # Apply StandardScaler
        normalized_data = self.scaler.fit_transform(self.df_cleaned)
        
        # Create DataFrame with normalized data
        self.df_normalized = pd.DataFrame(
            normalized_data, 
            columns=[f'{col}_normalized' for col in self.df_cleaned.columns],
            index=self.df_cleaned.index
        )
        
        print("Normalized data statistics:")
        print(self.df_normalized.describe())
        
        print("\nFirst 5 rows of normalized data:")
        print(self.df_normalized.head())
        
        return self.df_normalized
    
    def visualize_histograms(self):
        """Plot histograms before and after normalization"""
        if self.df_cleaned is None or self.df_normalized is None:
            print("Please complete outlier detection and normalization first!")
            return
        
        print("\n" + "="*50)
        print("HISTOGRAM VISUALIZATION")
        print("="*50)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Distribution of Confirmed and New Cases (Before and After Normalization)', fontsize=16)
        
        # Before normalization
        sns.histplot(data=self.df_cleaned, x='Confirmed', kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Confirmed Cases (Before Normalization)')
        axes[0, 0].set_xlabel('Confirmed Cases')
        
        sns.histplot(data=self.df_cleaned, x='New cases', kde=True, ax=axes[0, 1])
        axes[0, 1].set_title('New Cases (Before Normalization)')
        axes[0, 1].set_xlabel('New Cases')
        
        # After normalization
        sns.histplot(data=self.df_normalized, x='Confirmed_normalized', kde=True, ax=axes[1, 0])
        axes[1, 0].set_title('Confirmed Cases (After Normalization)')
        axes[1, 0].set_xlabel('Confirmed Cases (Normalized)')
        
        sns.histplot(data=self.df_normalized, x='New cases_normalized', kde=True, ax=axes[1, 1])
        axes[1, 1].set_title('New Cases (After Normalization)')
        axes[1, 1].set_xlabel('New Cases (Normalized)')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_correlation_heatmap(self):
        """Plot correlation heatmap between Confirmed and New cases"""
        if self.df is None:
            print("Please load the data first!")
            return
        
        print("\n" + "="*50)
        print("CORRELATION HEATMAP")
        print("="*50)
        
        # Calculate correlation matrix
        correlation_matrix = self.df.corr()
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={'shrink': 0.8})
            
        plt.title('Correlation Heatmap: Confirmed vs New Cases')
        plt.tight_layout()
        plt.show()
        
        return correlation_matrix
    
    def run_complete_analysis(self):
        """Run the complete EDA analysis pipeline"""
        print("STARTING COVID-19 EDA ANALYSIS")
        print("="*50)
        
        # Step 1: Load data
        if not self.load_data():
            return
        
        # Step 2: Compute statistics
        self.compute_statistics()
        
        # Step 3: Outlier detection
        self.detect_outliers_iqr()
        
        # Step 4: Normalization
        self.normalize_data()
        
        # Step 5: Visualizations
        self.visualize_histograms()
        self.visualize_correlation_heatmap()
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")

        print("="*50)

# Usage example
if __name__ == "__main__":
    # Create instance and run analysis
    covid_eda = CovidEDA('country_wise_latest.csv')
    covid_eda.run_complete_analysis()