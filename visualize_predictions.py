"""
Simple Visualization of Chlorophyll Predictions
Clean implementation for dataset visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Load and analyze the generated dataset and predictions"""
    print("CHLOROPHYLL PREDICTION SYSTEM - VISUALIZATION")
    print("=" * 50)
    
    # Load datasets
    timeseries_df = pd.read_csv('chl_connect_timeseries_2000pts.csv')
    predictions_df = pd.read_csv('enhanced_chl_connect_predictions.csv')
    
    print(f"📊 Time Series Dataset:")
    print(f"   Total observations: {len(timeseries_df):,}")
    print(f"   Unique pixels: {timeseries_df[['lat', 'lon']].drop_duplicates().shape[0]}")
    print(f"   Date range: {timeseries_df['date'].min()} to {timeseries_df['date'].max()}")
    print(f"   Chlorophyll range: {timeseries_df['chlorophyll_a'].min():.1f} - {timeseries_df['chlorophyll_a'].max():.1f} mg/m³")
    
    print(f"\n🔮 Predictions Dataset:")
    print(f"   Future predictions: {len(predictions_df):,}")
    print(f"   Prediction pixels: {predictions_df[['pixel_x', 'pixel_y']].drop_duplicates().shape[0]}")
    print(f"   Forecast dates: {predictions_df['date'].nunique()}")
    print(f"   Predicted range: {predictions_df['chlorophyll'].min():.1f} - {predictions_df['chlorophyll'].max():.1f} mg/m³")
    
    return timeseries_df, predictions_df

def create_scatter_maps(timeseries_df, predictions_df):
    """Create simple scatter plot maps showing pixel locations and chlorophyll levels"""
    
    # Get sample dates
    current_dates = sorted(timeseries_df['date'].unique())[-3:]  # Last 3 dates
    future_dates = sorted(predictions_df['date'].unique())[:3]   # First 3 future dates
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Current data (top row)
    for i, date in enumerate(current_dates):
        ax = axes[0, i]
        data = timeseries_df[timeseries_df['date'] == date]
        
        scatter = ax.scatter(data['lon'], data['lat'], 
                           c=data['chlorophyll_a'], 
                           cmap='RdYlGn_r', s=60, alpha=0.8,
                           vmin=0, vmax=100)
        
        ax.set_title(f'Current: {date}\n{len(data)} pixels')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Chl-a (mg/m³)', fontsize=10)
    
    # Future predictions (bottom row)
    for i, date in enumerate(future_dates):
        ax = axes[1, i]
        data = predictions_df[predictions_df['date'] == date]
        
        scatter = ax.scatter(data['pixel_x'], data['pixel_y'], 
                           c=data['chlorophyll'], 
                           cmap='RdYlGn_r', s=60, alpha=0.8,
                           vmin=0, vmax=100)
        
        ax.set_title(f'Predicted: {date}\n{len(data)} pixels')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Chl-a (mg/m³)', fontsize=10)
    
    plt.suptitle('Chlorophyll Distribution: Current vs Predicted\nTrês Marias Reservoir', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('chlorophyll_scatter_maps.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Scatter maps saved: chlorophyll_scatter_maps.png")

def create_temporal_analysis(timeseries_df, predictions_df):
    """Create temporal analysis plots"""
    
    # Aggregate by date for trends
    current_trend = timeseries_df.groupby('date')['chlorophyll_a'].agg(['mean', 'std', 'count']).reset_index()
    future_trend = predictions_df.groupby('date')['chlorophyll'].agg(['mean', 'std', 'count']).reset_index()
    
    # Convert dates
    current_trend['date'] = pd.to_datetime(current_trend['date'])
    future_trend['date'] = pd.to_datetime(future_trend['date'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Temporal trend
    ax1.errorbar(current_trend['date'], current_trend['mean'], 
                yerr=current_trend['std'], fmt='o-', color='blue', 
                label='Historical (CHL-CONNECT)', alpha=0.8)
    
    ax1.errorbar(future_trend['date'], future_trend['mean'], 
                yerr=future_trend['std'], fmt='s--', color='red', 
                label='Predicted (ML Model)', alpha=0.8)
    
    ax1.set_ylabel('Chlorophyll-a (mg/m³)')
    ax1.set_title('Temporal Chlorophyll Trend: Historical vs Predicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution comparison
    ax2.hist(timeseries_df['chlorophyll_a'], bins=30, alpha=0.6, 
             label=f'Historical (n={len(timeseries_df)})', color='blue', density=True)
    ax2.hist(predictions_df['chlorophyll'], bins=30, alpha=0.6, 
             label=f'Predicted (n={len(predictions_df)})', color='red', density=True)
    
    ax2.set_xlabel('Chlorophyll-a (mg/m³)')
    ax2.set_ylabel('Density')
    ax2.set_title('Chlorophyll Distribution: Historical vs Predicted')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Temporal analysis saved: temporal_analysis.png")

def create_summary_statistics(timeseries_df, predictions_df):
    """Create summary statistics table"""
    
    stats = {
        'Dataset': ['Historical Time Series', 'Future Predictions'],
        'Observations': [len(timeseries_df), len(predictions_df)],
        'Unique Pixels': [
            timeseries_df[['lat', 'lon']].drop_duplicates().shape[0],
            predictions_df[['pixel_x', 'pixel_y']].drop_duplicates().shape[0]
        ],
        'Date Range': [
            f"{timeseries_df['date'].min()} to {timeseries_df['date'].max()}",
            f"{predictions_df['date'].min()} to {predictions_df['date'].max()}"
        ],
        'Chl-a Mean (mg/m³)': [
            f"{timeseries_df['chlorophyll_a'].mean():.1f}",
            f"{predictions_df['chlorophyll'].mean():.1f}"
        ],
        'Chl-a Range (mg/m³)': [
            f"{timeseries_df['chlorophyll_a'].min():.1f} - {timeseries_df['chlorophyll_a'].max():.1f}",
            f"{predictions_df['chlorophyll'].min():.1f} - {predictions_df['chlorophyll'].max():.1f}"
        ]
    }
    
    stats_df = pd.DataFrame(stats)
    
    print("\n📋 SUMMARY STATISTICS:")
    print(stats_df.to_string(index=False))
    
    # Save to CSV
    stats_df.to_csv('summary_statistics.csv', index=False)
    print("\n✅ Summary statistics saved: summary_statistics.csv")

def main():
    """Main visualization pipeline"""
    
    # Load and analyze data
    timeseries_df, predictions_df = load_and_analyze_data()
    
    # Create visualizations
    print("\n🗺️ Creating scatter maps...")
    create_scatter_maps(timeseries_df, predictions_df)
    
    print("\n📈 Creating temporal analysis...")
    create_temporal_analysis(timeseries_df, predictions_df)
    
    # Create summary
    create_summary_statistics(timeseries_df, predictions_df)
    
    print("\n🎉 VISUALIZATION COMPLETE!")
    print("Generated files:")
    print("  - chlorophyll_scatter_maps.png")
    print("  - temporal_analysis.png") 
    print("  - summary_statistics.csv")

if __name__ == "__main__":
    main()