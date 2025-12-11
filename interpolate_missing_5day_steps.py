import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timedelta

def interpolate_5day_gaps():
    # Setup directories
    input_dir = "daily_snapshots"
    output_dir = "daily_snapshots_5day"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Get all existing files
    csv_files = sorted(glob.glob(os.path.join(input_dir, "snapshot_*.csv")))
    print(f"Found {len(csv_files)} original snapshot files")
    
    # Read all files into a single DataFrame
    dfs = []
    for f in csv_files:
        # Extract date from filename
        date_str = os.path.basename(f).replace("snapshot_", "").replace(".csv", "")
        date = pd.to_datetime(date_str)
        
        df = pd.read_csv(f)
        df['date'] = date
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(full_df)} rows of data")
    
    # Handle duplicates (same grid_id on same date)
    # This was an issue before, so we keep this safety measure
    full_df = full_df.groupby(['date', 'grid_id']).mean(numeric_only=True).reset_index()
    
    # Pivot to wide format: Index=Date, Columns=GridID, Values=Features
    # We need to interpolate all features
    features = [c for c in full_df.columns if c not in ['date', 'grid_id']]
    print(f"Features to interpolate: {features}")
    
    # Create a complete 5-day date range
    start_date = full_df['date'].min()
    end_date = full_df['date'].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='5D')
    print(f"Target date range: {start_date.date()} to {end_date.date()} (Every 5 days)")
    print(f"Total expected steps: {len(date_range)}")
    
    # Process each feature
    interpolated_dfs = []
    
    for feature in features:
        print(f"Processing feature: {feature}")
        # Pivot
        pivot_df = full_df.pivot(index='date', columns='grid_id', values=feature)
        
        # Reindex to the 5-day range
        pivot_df = pivot_df.reindex(date_range)
        
        # Interpolate (Time-based linear interpolation)
        pivot_df_interp = pivot_df.interpolate(method='time')
        
        # Unstack back to long format
        long_df = pivot_df_interp.unstack().reset_index()
        long_df.columns = ['grid_id', 'date', feature]
        
        if len(interpolated_dfs) == 0:
            interpolated_dfs.append(long_df)
        else:
            # Merge with existing results
            interpolated_dfs[0] = pd.merge(interpolated_dfs[0], long_df, on=['date', 'grid_id'])
            
    final_df = interpolated_dfs[0]
    
    # Save files
    count = 0
    for date in date_range:
        day_data = final_df[final_df['date'] == date].copy()
        
        # Format date column to string YYYY-MM-DD
        day_data['date'] = day_data['date'].dt.strftime('%Y-%m-%d')
        
        # Reorder columns to match original if possible, or at least keep grid_id first
        # Ensure 'date' is included
        cols = ['grid_id', 'date'] + [c for c in day_data.columns if c not in ['grid_id', 'date']]
        day_data_save = day_data[cols]
        
        date_str = date.strftime('%Y-%m-%d')
        filename = f"snapshot_{date_str}.csv"
        output_path = os.path.join(output_dir, filename)
        
        day_data_save.to_csv(output_path, index=False)
        count += 1
        
    print(f"Successfully generated {count} files in {output_dir}")

if __name__ == "__main__":
    interpolate_5day_gaps()
