from covid_analysis import COVIDAnalysis
from covid_visualization import CovidVisualization
import os

def main():
    # Try different possible file paths
    possible_paths = [
        'country_wise_latest.csv',  # Same directory as main.py
        '../country_wise_latest.csv',  # Parent directory
        './CovidProject/country_wise_latest.csv',  # Subdirectory
        'Week5/CovidProject/country_wise_latest.csv',  # Full path from current
        r'C:\Users\Janani\Documents\AI Engineer\Programming\Python\code\Week5\CovidProject\country_wise_latest.csv'  # Absolute path
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            print(f"Found file at: {path}")
            break
    
    if file_path is None:
        print("CSV file not found in any common locations.")
        print("Current working directory:", os.getcwd())
        print("Please check if 'country_wise_latest.csv' exists in your project folder.")
        return

    # Run numerical analysis
    print("=" * 60)
    print("RUNNING NUMERICAL ANALYSIS")
    print("=" * 60)
    
    analyzer = COVIDAnalysis(file_path)
    if analyzer.df is not None:
        print("\nDataset Overview:")
        print(analyzer.display_head())
        results = analyzer.run_all_analyses()
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
    else:
        print("Failed to load data for analysis.")
        return
    
    # Run visualizations
    print("\n" + "=" * 60)
    print("RUNNING VISUALIZATIONS")
    print("=" * 60)
    
    visualizer = CovidVisualization(file_path)
    if visualizer.df is not None:
        visualization_results = visualizer.run_all_visualizations()
        print("\n" + "="*60)
        print("ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
    else:
        print("Failed to load data for visualizations.")

if __name__ == "__main__":
    main()