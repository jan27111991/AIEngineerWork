from covid_base import COVIDDataAnalyzer

class COVIDAnalysis(COVIDDataAnalyzer):
    """Child class that inherits from COVIDDataAnalyzer and implements specific analyses"""
    
    def __init__(self, file_path):
        super().__init__(file_path)
    
    def summarize_cases_by_region(self):
        """1. Summarize Case Counts by Region"""
        if self.df is not None:
            regional_summary = self.df.groupby('WHO Region').agg({
                'Confirmed': 'sum',
                'Deaths': 'sum',
                'Recovered': 'sum'
            }).reset_index()
            
            print("=== Case Counts by Region ===")
            print(regional_summary)
            return regional_summary
    
    def filter_low_case_records(self):
        """2. Filter Low Case Records (Confirmed cases < 10)"""
        if self.df is not None:
            filtered_df = self.df[self.df['Confirmed'] >= 10]
            print(f"\n=== Filtered Data (Confirmed >= 10) ===")
            print(f"Original records: {len(self.df)}")
            print(f"Filtered records: {len(filtered_df)}")
            return filtered_df
    
    def region_highest_confirmed_cases(self):
        """3. Identify Region with Highest Confirmed Cases"""
        if self.df is not None:
            region_cases = self.df.groupby('WHO Region')['Confirmed'].sum()
            max_region = region_cases.idxmax()
            max_cases = region_cases.max()
            
            print(f"\n=== Region with Highest Confirmed Cases ===")
            print(f"Region: {max_region}")
            print(f"Total Confirmed Cases: {max_cases:,}")
            return max_region, max_cases
    
    def sort_by_confirmed_cases(self, output_file='sorted_covid_data.csv'):
        """4. Sort Data by Confirmed Cases and save to CSV"""
        if self.df is not None:
            sorted_df = self.df.sort_values('Confirmed', ascending=False)
            sorted_df.to_csv(output_file, index=False)
            print(f"\n=== Data Sorted by Confirmed Cases ===")
            print(f"Sorted data saved to: {output_file}")
            print(f"Top 5 countries by confirmed cases:")
            print(sorted_df[['Country/Region', 'Confirmed']].head())
            return sorted_df
    
    def top_5_countries_by_cases(self):
        """5. Top 5 Countries by Case Count"""
        if self.df is not None:
            top_countries = self.df.nlargest(5, 'Confirmed')[['Country/Region', 'Confirmed']]
            print("\n=== Top 5 Countries by Confirmed Cases ===")
            for idx, row in top_countries.iterrows():
                print(f"{row['Country/Region']}: {row['Confirmed']:,} cases")
            return top_countries
    
    def region_lowest_death_count(self):
        """6. Region with Lowest Death Count"""
        if self.df is not None:
            region_deaths = self.df.groupby('WHO Region')['Deaths'].sum()
            min_region = region_deaths.idxmin()
            min_deaths = region_deaths.min()
            
            print(f"\n=== Region with Lowest Death Count ===")
            print(f"Region: {min_region}")
            print(f"Total Deaths: {min_deaths:,}")
            return min_region, min_deaths
    
    def india_case_summary(self):
        """7. India's Case Summary"""
        if self.df is not None:
            india_data = self.df[self.df['Country/Region'] == 'India']
            if not india_data.empty:
                india_row = india_data.iloc[0]
                print("\n=== India's COVID-19 Case Summary ===")
                print(f"Confirmed Cases: {india_row['Confirmed']:,}")
                print(f"Deaths: {india_row['Deaths']:,}")
                print(f"Recovered: {india_row['Recovered']:,}")
                print(f"Active Cases: {india_row['Active']:,}")
                print(f"Mortality Rate: {india_row['Deaths / 100 Cases']}%")
                print(f"Recovery Rate: {india_row['Recovered / 100 Cases']}%")
                return india_row
            else:
                print("India data not found in the dataset.")
                return None
    
    def calculate_mortality_rate_by_region(self):
        """8. Calculate Mortality Rate by Region (Death-to-confirmed case ratio)"""
        if self.df is not None:
            regional_data = self.df.groupby('WHO Region').agg({
                'Confirmed': 'sum',
                'Deaths': 'sum'
            })
            regional_data['Mortality Rate (%)'] = (regional_data['Deaths'] / regional_data['Confirmed']) * 100
            
            print("\n=== Mortality Rate by Region ===")
            for region, row in regional_data.iterrows():
                print(f"{region}: {row['Mortality Rate (%)']:.2f}%")
            return regional_data
    
    def compare_recovery_rates(self):
        """9. Compare Recovery Rates Across Regions"""
        if self.df is not None:
            regional_data = self.df.groupby('WHO Region').agg({
                'Confirmed': 'sum',
                'Recovered': 'sum'
            })
            regional_data['Recovery Rate (%)'] = (regional_data['Recovered'] / regional_data['Confirmed']) * 100
            
            print("\n=== Recovery Rate by Region ===")
            recovery_rates_sorted = regional_data.sort_values('Recovery Rate (%)', ascending=False)
            for region, row in recovery_rates_sorted.iterrows():
                print(f"{region}: {row['Recovery Rate (%)']:.2f}%")
            return recovery_rates_sorted
    
    def detect_outliers_in_case_counts(self):
        """10. Detect Outliers in Case Counts using mean ± 2*std deviation"""
        if self.df is not None:
            confirmed_cases = self.df['Confirmed']
            mean_cases = confirmed_cases.mean()
            std_cases = confirmed_cases.std()
            
            lower_bound = mean_cases - 2 * std_cases
            upper_bound = mean_cases + 2 * std_cases
            
            outliers = self.df[(self.df['Confirmed'] < lower_bound) | (self.df['Confirmed'] > upper_bound)]
            
            print("\n=== Outliers in Confirmed Cases (Mean ± 2*STD) ===")
            print(f"Mean: {mean_cases:.2f}")
            print(f"Standard Deviation: {std_cases:.2f}")
            print(f"Lower Bound: {lower_bound:.2f}")
            print(f"Upper Bound: {upper_bound:.2f}")
            print(f"Number of outliers: {len(outliers)}")
            print("Outlier countries:")
            for idx, row in outliers.iterrows():
                print(f"{row['Country/Region']}: {row['Confirmed']:,} cases")
            
            return outliers
    
    def group_by_country_region(self):
        """11. Group Data by Country and Region"""
        if self.df is not None:
            grouped_data = self.df.groupby(['WHO Region', 'Country/Region']).agg({
                'Confirmed': 'first',
                'Deaths': 'first',
                'Recovered': 'first'
            })
            print("\n=== Data Grouped by Region and Country ===")
            print("Grouping completed. Use the returned object for further analysis.")
            return grouped_data
    
    def regions_with_zero_recovered_cases(self):
        """12. Identify Regions with Zero Recovered Cases"""
        if self.df is not None:
            zero_recovered_countries = self.df[self.df['Recovered'] == 0]
            regions_with_zero = zero_recovered_countries['WHO Region'].unique()
            
            print("\n=== Regions with Countries Having Zero Recovered Cases ===")
            if len(regions_with_zero) > 0:
                for region in regions_with_zero:
                    countries_in_region = zero_recovered_countries[zero_recovered_countries['WHO Region'] == region]
                    print(f"\n{region}:")
                    for idx, row in countries_in_region.iterrows():
                        print(f"  - {row['Country/Region']} (Confirmed: {row['Confirmed']})")
            else:
                print("No regions found with countries having zero recovered cases.")
            
            return regions_with_zero, zero_recovered_countries
    
    def run_all_analyses(self):
        """Run all required analyses"""
        print("=" * 60)
        print("COVID-19 DATA ANALYSIS REPORT")
        print("=" * 60)
        
        analyses = [
            self.summarize_cases_by_region,
            self.filter_low_case_records,
            self.region_highest_confirmed_cases,
            self.sort_by_confirmed_cases,
            self.top_5_countries_by_cases,
            self.region_lowest_death_count,
            self.india_case_summary,
            self.calculate_mortality_rate_by_region,
            self.compare_recovery_rates,
            self.detect_outliers_in_case_counts,
            self.group_by_country_region,
            self.regions_with_zero_recovered_cases
        ]
        
        results = {}
        for i, analysis in enumerate(analyses, 1):
            print(f"\n{'='*50}")
            print(f"ANALYSIS {i}: {analysis.__name__.replace('_', ' ').title()}")
            print(f"{'='*50}")
            try:
                result = analysis()
                results[analysis.__name__] = result
            except Exception as e:
                print(f"Error in analysis: {e}")
        
        return results