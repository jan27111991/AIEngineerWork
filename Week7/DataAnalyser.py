import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    def __init__(self, data_frame):
        self.data = data_frame
        self.analysis_results = {}
    
    def calculate_basic_stats(self):
        """Calculate basic statistics for numeric columns"""
        numeric_cols = ['R_Priority', 'Complexity', 'Time', 'Cost']
        available_cols = [col for col in numeric_cols if col in self.data.columns]
        
        stats = self.data[available_cols].describe()
        return stats
    
    def calculate_correlations(self):
        """Calculate correlation matrix for numeric columns"""
        numeric_cols = ['R_Priority', 'Complexity', 'Time', 'Cost']
        available_cols = [col for col in numeric_cols if col in self.data.columns]
        
        if len(available_cols) < 2:
            print("Not enough numeric columns for correlation analysis")
            return None
            
        correlation_matrix = self.data[available_cols].corr()
        return correlation_matrix
    
    def plot_correlation_heatmap(self):
        """Create a heatmap visualization of correlations"""
        corr_matrix = self.calculate_correlations()
        if corr_matrix is None:
            return
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='RdYlBu', 
                   vmin=-1, vmax=1,
                   square=True,
                   fmt='.2f')
        plt.title('Test Metrics Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def plot_distributions(self):
        """Plot distribution histograms for key metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        metrics = ['Complexity', 'Time', 'R_Priority', 'Cost']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            if metric in self.data.columns:
                axes[i].hist(self.data[metric], bins=15, alpha=0.7, color=color, edgecolor='black')
                axes[i].set_title(f'{metric} Distribution')
                axes[i].set_xlabel(metric)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Hide any unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    
    def generate_analysis_report(self):
        """Generate a comprehensive analysis report"""
        print("=== DATA ANALYSIS REPORT ===")
        
        # Basic stats
        stats = self.calculate_basic_stats()
        print("\nBasic Statistics:")
        print(stats)
        
        # Correlations
        corr_matrix = self.calculate_correlations()
        if corr_matrix is not None:
            print("\nCorrelation Matrix:")
            print(corr_matrix)
            
            # Find strongest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.3:  # Only show meaningful correlations
                        corr_pairs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_val
                        ))
            
            if corr_pairs:
                print("\nNotable Correlations (|r| > 0.3):")
                for col1, col2, corr in corr_pairs:
                    print(f"  {col1} vs {col2}: {corr:.3f}")
        
        return {
            'statistics': stats,
            'correlations': corr_matrix
        }