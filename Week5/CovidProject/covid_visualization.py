import matplotlib
matplotlib.use('Agg')  # Use Agg backend

# Import specific matplotlib components
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from covid_analysis import COVIDAnalysis

class CovidVisualization(COVIDAnalysis):
    """Child class that inherits from COVIDAnalysis and adds visualization capabilities"""
    
    def __init__(self, file_path):
        super().__init__(file_path)
        # Set up matplotlib style using rcParams
        matplotlib.rcParams.update({
            'figure.figsize': [10, 6],
            'font.size': 12,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']
    
    def bar_chart_top_10_countries(self):
        """1. Bar Chart of Top 10 Countries by Confirmed Cases"""
        if self.df is not None:
            top_10 = self.df.nlargest(10, 'Confirmed')[['Country/Region', 'Confirmed']]
            
            fig = Figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            bars = ax.bar(top_10['Country/Region'], top_10['Confirmed'], 
                         color=self.colors, alpha=0.8, edgecolor='black')
            """
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 10000,
                       f'{height:,}', ha='center', va='bottom', fontsize=10)
            """
            ax.set_title('Top 10 Countries by Confirmed COVID-19 Cases', fontsize=16, fontweight='bold')
            ax.set_xlabel('Countries', fontsize=12)
            ax.set_ylabel('Confirmed Cases', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, axis='y', alpha=0.3)
            fig.tight_layout()
            
            # Save the figure
            fig.savefig('bar_chart_top_10.png', dpi=100, bbox_inches='tight')
            print("✓ Bar chart saved as 'bar_chart_top_10.png'")
            
            return top_10
    
    def pie_chart_global_death_distribution(self):
        """2. Pie Chart of Global Death Distribution by Region"""
        if self.df is not None:
            death_by_region = self.df.groupby('WHO Region')['Deaths'].sum()
            
            fig = Figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            wedges, texts, autotexts = ax.pie(death_by_region.values, 
                                            labels=death_by_region.index,
                                            autopct='%1.1f%%',
                                            colors=self.colors,
                                            startangle=90)
            
            # Style the text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('Global COVID-19 Death Distribution by WHO Region', 
                        fontsize=16, fontweight='bold')
            ax.axis('equal')
            fig.tight_layout()
            
            # Save the figure
            fig.savefig('pie_chart_deaths.png', dpi=100, bbox_inches='tight')
            print("✓ Pie chart saved as 'pie_chart_deaths.png'")
            
            return death_by_region
    
    def line_chart_comparison_top_5(self):
        """3. Line Chart comparing Confirmed Cases and Deaths for Top 5 Countries"""
        if self.df is not None:
            top_5 = self.df.nlargest(5, 'Confirmed')[['Country/Region', 'Confirmed', 'Deaths']]
        
            fig = Figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
        
            # Create x-axis positions (0, 1, 2, 3, 4 for 5 countries)
            x_pos = np.arange(len(top_5))
        
        # Create line plots 
            ax.plot(x_pos, top_5['Confirmed'], 
                marker='o', 
                linewidth=3, 
                markersize=8,
                label='Confirmed Cases', 
                color='#FF6B6B', 
                alpha=0.8)
        
            ax.plot(x_pos, top_5['Deaths'], 
                marker='s', 
                linewidth=3, 
                markersize=8,
                label='Deaths', 
                color='#4ECDC4', 
                alpha=0.8)
        
        # Customize the chart
            ax.set_title('Comparison of Confirmed Cases vs Deaths for Top 5 Countries', 
                    fontsize=14, fontweight='bold')
            ax.set_xlabel('Countries', fontsize=12)
            ax.set_ylabel('Number of Cases', fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(top_5['Country/Region'], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Add value annotations on each point
        for i, (confirmed, death) in enumerate(zip(top_5['Confirmed'], top_5['Deaths'])):
            ax.annotate(f'{confirmed:,}', 
                       (x_pos[i], confirmed), 
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center', 
                       fontsize=9,
                       color='#FF6B6B')
            
            ax.annotate(f'{death:,}', 
                       (x_pos[i], death), 
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center', 
                       fontsize=9,
                       color='#4ECDC4')
        
        fig.tight_layout()
        
        # Save the figure
        fig.savefig('line_chart_comparison_top_5.png', dpi=100, bbox_inches='tight')
        print("✓ Line comparison chart saved as 'line_chart_comparison_top_5.png'")
        
        return top_5
    
    def scatter_plot_confirmed_vs_recovered(self):
        """4. Scatter Plot comparing Confirmed and Recovered Cases"""
        if self.df is not None:
            fig = Figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Filter out countries with very low cases
        filtered_df = self.df[self.df['Confirmed'] > 1000]
        
        # Create scatter plot for Confirmed Cases (red)
        ax.scatter(filtered_df['Confirmed'], 
                  alpha=0.7, s=60, color='red', label='Confirmed Cases')
        
        # Create scatter plot for Recovered Cases (purple)
        ax.scatter(filtered_df['Confirmed'],
                  alpha=0.7, s=60, color='purple', label='Recovered Cases')
        
        ax.set_title('Confirmed Cases vs Recovery Comparison', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Confirmed Cases', fontsize=12)
        ax.set_ylabel('Number of Cases', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        # Save the figure
        fig.savefig('scatter_plot.png', dpi=100, bbox_inches='tight')
        print("✓ Scatter plot saved as 'scatter_plot.png'")
        
        return filtered_df[['Country/Region', 'Confirmed', 'Recovered']]
    def histogram_death_counts(self):
        """5. Histogram of Death Counts across all Regions"""
        if self.df is not None:
            fig = Figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            
            deaths_data = self.df[self.df['Deaths'] > 0]['Deaths']
            
            ax.hist(deaths_data, bins=30, color='#FF6B6B', alpha=0.8, 
                   edgecolor='black')
            
            ax.set_title('Distribution of COVID-19 Death Counts Across Countries', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Number of Deaths', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            
            # Save the figure
            fig.savefig('histogram_deaths.png', dpi=100, bbox_inches='tight')
            print("✓ Histogram saved as 'histogram_deaths.png'")
            
            return deaths_data.describe()
    
    def stacked_bar_chart_selected_countries(self):
        """6. Stacked Bar Chart of Confirmed, Deaths, and Recovered for 5 Selected Countries"""
        if self.df is not None:
            # Select diverse countries for better representation
            selected_countries = ['US', 'India', 'Brazil', 'Russia', 'South Africa']
            selected_data = self.df[self.df['Country/Region'].isin(selected_countries)]
            
            # Calculate active cases
            selected_data = selected_data.copy()
            selected_data['Active'] = selected_data['Confirmed'] - selected_data['Deaths'] - selected_data['Recovered']
            
            # Prepare data for stacking
            categories = ['Deaths', 'Recovered', 'Active']
            bottom = np.zeros(len(selected_data))
            
            fig = Figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Deaths, Recovered, Active
            
            for i, category in enumerate(categories):
                values = selected_data[category].values
                ax.bar(selected_data['Country/Region'], values, bottom=bottom, 
                      label=category, color=colors[i], alpha=0.8, edgecolor='black')
                bottom += values
            
            ax.set_title('COVID-19 Case Distribution: Deaths, Recovered, and Active Cases\n(Selected Countries)', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Countries', fontsize=12)
            ax.set_ylabel('Number of Cases', fontsize=12)
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
            fig.tight_layout()
            
            # Save the figure
            fig.savefig('stacked_bar_chart.png', dpi=100, bbox_inches='tight')
            print("✓ Stacked bar chart saved as 'stacked_bar_chart.png'")
            
            return selected_data[['Country/Region', 'Confirmed', 'Deaths', 'Recovered', 'Active']]
    
    def box_plot_confirmed_by_region(self):
        """7. Box Plot of Confirmed Cases across Regions"""
        if self.df is not None:
            fig = Figure(figsize=(14, 8))
            ax = fig.add_subplot(111)
            
            # Prepare data for boxplot
            regions_data = []
            regions_labels = []
            
            for region in self.df['WHO Region'].unique():
                region_cases = self.df[self.df['WHO Region'] == region]['Confirmed']
                regions_data.append(region_cases)
                regions_labels.append(region)
            
            # Create boxplot
            boxplot = ax.boxplot(regions_data, labels=regions_labels, patch_artist=True)
            
            # Color the boxes manually
            for i, patch in enumerate(boxplot['boxes']):
                patch.set_facecolor(self.colors[i % len(self.colors)])
                patch.set_alpha(0.7)
            
            ax.set_title('Distribution of Confirmed COVID-19 Cases Across WHO Regions', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('WHO Regions', fontsize=12)
            ax.set_ylabel('Confirmed Cases', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, axis='y', alpha=0.3)
            fig.tight_layout()
            
            # Save the figure
            fig.savefig('box_plot_regions.png', dpi=100, bbox_inches='tight')
            print("✓ Box plot saved as 'box_plot_regions.png'")
            
            # Return summary statistics
            return self.df.groupby('WHO Region')['Confirmed'].describe()
    
    def trend_line_india_comparison(self, compare_country='US'):
        """8. Bar Chart comparison: India vs another chosen country"""
        if self.df is not None:
            # Get data for India and comparison country
            india_data = self.df[self.df['Country/Region'] == 'India']
            compare_data = self.df[self.df['Country/Region'] == compare_country]
            
            if not india_data.empty and not compare_data.empty:
                india_row = india_data.iloc[0]
                compare_row = compare_data.iloc[0]
                
                # Create comparison data
                countries = ['India', compare_country]
                confirmed_cases = [india_row['Confirmed'], compare_row['Confirmed']]
                deaths = [india_row['Deaths'], compare_row['Deaths']]
                
                fig = Figure(figsize=(15, 6))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                
                # Plot 1: Confirmed Cases Comparison
                bars1 = ax1.bar(countries, confirmed_cases, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
                ax1.set_title('Confirmed Cases Comparison', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Number of Cases', fontsize=12)
                ax1.grid(True, axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar in bars1:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 10000,
                            f'{height:,}', ha='center', va='bottom', fontsize=11)
                
                # Plot 2: Deaths Comparison
                bars2 = ax2.bar(countries, deaths, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
                ax2.set_title('Deaths Comparison', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Number of Deaths', fontsize=12)
                ax2.grid(True, axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar in bars2:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 1000,
                            f'{height:,}', ha='center', va='bottom', fontsize=11)
                
                fig.suptitle(f'COVID-19 Comparison: India vs {compare_country}', 
                           fontsize=16, fontweight='bold')
                fig.tight_layout()
                
                # Save the figure
                fig.savefig(f'comparison_india_vs_{compare_country}.png', dpi=100, bbox_inches='tight')
                print(f"✓ Comparison chart saved as 'comparison_india_vs_{compare_country}.png'")
                
                comparison_data = {
                    'Country': countries,
                    'Confirmed': confirmed_cases,
                    'Deaths': deaths
                }
                
                print(f"\nComparison Summary:")
                print(f"India - Confirmed: {india_row['Confirmed']:,}, Deaths: {india_row['Deaths']:,}")
                print(f"{compare_country} - Confirmed: {compare_row['Confirmed']:,}, Deaths: {compare_row['Deaths']:,}")
                
                return pd.DataFrame(comparison_data)
            else:
                print(f"Data for India or {compare_country} not found.")
                return None
    
    def run_all_visualizations(self):
        """Run all visualization analyses"""
        print("=" * 60)
        print("COVID-19 DATA VISUALIZATION REPORT")
        print("=" * 60)
        
        visualizations = [
            self.bar_chart_top_10_countries,
            self.pie_chart_global_death_distribution,
            self.line_chart_comparison_top_5,
            self.scatter_plot_confirmed_vs_recovered,
            self.histogram_death_counts,
            self.stacked_bar_chart_selected_countries,
            self.box_plot_confirmed_by_region,
            lambda: self.trend_line_india_comparison('US')
        ]
        
        results = {}
        for i, viz in enumerate(visualizations, 1):
            print(f"\n{'='*50}")
            print(f"VISUALIZATION {i}: {viz.__name__.replace('_', ' ').title()}")
            print(f"{'='*50}")
            try:
                result = viz()
                results[viz.__name__] = result
            except Exception as e:
                print(f"Error in visualization: {e}")
        
        print(f"\nAll visualizations have been saved as PNG files in the current directory.")
        return results