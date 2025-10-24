from DataCleaner import DataCleaner
from DataAnalyser import DataAnalyzer
from TimePredictor import TimePredictor

def main():
    print("=== TEST DATA ANALYSIS TOOL ===\n")
    
    # Step 1: Load and clean data
    print("1. Loading and cleaning data...")
    cleaner = DataCleaner('Testing_carlease.csv')
    
    if not cleaner.load_data():
        print("Failed to load data. Exiting.")
        return
    
    cleaned_data = cleaner.clean_all_data()
    cleaner.get_cleaning_summary()
    
    # Step 2: Analyze the cleaned data
    print("\n2. Analyzing cleaned data...")
    analyzer = DataAnalyzer(cleaned_data)
    analysis_report = analyzer.generate_analysis_report()
    
    # Create visualizations
    analyzer.plot_correlation_heatmap()
    analyzer.plot_distributions()
    
    # Step 3: Build prediction model
    print("\n3. Building time prediction model...")
    predictor = TimePredictor(cleaned_data)
    model_results = predictor.run_complete_analysis()
    
    # Final summary
    print("\n=== ANALYSIS COMPLETE ===")
    print("All tasks completed successfully!")
    print(f"Final dataset size: {cleaned_data.shape}")
    
    if model_results:
        print(f"Model performance - R²: {model_results['r2']:.3f}")

if __name__ == "__main__":
    main()