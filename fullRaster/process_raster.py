"""
Process Exported Rasters - Apply Chl-CONNECT to Full Images
Reads GeoTIFF files and calculates chlorophyll-a for every pixel

Requirements:
    pip install rasterio numpy pandas --break-system-packages

Author: Water Quality Remote Sensing
"""

import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
from tqdm import tqdm

# Add CHL-CONNECT to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Chl-CONNECT'))

class RasterProcessor:
    def __init__(self, input_dir='downloaded_rasters', output_dir='chlorophyll_rasters'):
        """
        Initialize the raster processor
        
        Args:
            input_dir: Directory containing downloaded GeoTIFF files
            output_dir: Directory to save chlorophyll rasters
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def list_raster_files(self):
        """
        List all GeoTIFF files in input directory
        """
        files = sorted(self.input_dir.glob('*.tif'))
        print(f"Found {len(files)} raster files")
        return files
    
    def process_single_raster(self, input_file, save_output=True):
        """
        Process a single raster file and calculate chlorophyll
        
        Args:
            input_file: Path to input GeoTIFF
            save_output: Whether to save output raster
            
        Returns:
            tuple: (chlorophyll_map, water_class_map, metadata)
        """
        print(f"\nProcessing: {input_file.name}")
        
        try:
            from common.Chl_CONNECT import Chl_CONNECT
        except ImportError:
            print("❌ CHL-CONNECT not available. Please install the library.")
            return None, None, None
        
        try:
            with rasterio.open(input_file) as src:
                # Read metadata
                meta = src.meta.copy()
                height, width = src.height, src.width
                transform = src.transform
                crs = src.crs
                
                print(f"   Dimensions: {width} x {height} pixels")
                print(f"   Total pixels: {width * height:,}")
                
                # Read Rrs bands (should be 8 bands but we only need 5 for MSI)
                # Band order: Rrs443, Rrs490, Rrs560, Rrs665, Rrs705, Rrs740, Rrs783, Rrs865
                rrs_data = src.read([1, 2, 3, 4, 5])  # Read first 5 bands
                
                # Get nodata mask (where mask is applied)
                valid_mask = src.read_masks(1) > 0  # Non-zero means valid data
                n_valid = valid_mask.sum()
                
                print(f"   Valid water pixels: {n_valid:,} ({100*n_valid/(width*height):.1f}%)")
                
                if n_valid == 0:
                    print("   ⚠️ No valid pixels found!")
                    return None, None, meta
                
                # Reshape for Chl-CONNECT: (n_valid_pixels, 5)
                rrs_flat = rrs_data[:, valid_mask].T  # Shape: (n_valid, 5)
                
                print(f"   Applying Chl-CONNECT to {n_valid:,} pixels...")
                
                # Apply Chl-CONNECT
                chl_conn = Chl_CONNECT(Rrs_input=rrs_flat, sensor='MSI')
                chl_values = chl_conn.Chl_comb
                water_classes = chl_conn.Class
                
                # Create output arrays (fill with nodata value)
                chl_map = np.full((height, width), -9999.0, dtype=np.float32)
                class_map = np.full((height, width), -9999, dtype=np.int16)
                
                # Fill in valid pixels
                chl_map[valid_mask] = chl_values
                class_map[valid_mask] = water_classes
                
                # Calculate statistics
                valid_chl = chl_values[~np.isnan(chl_values)]
                if len(valid_chl) > 0:
                    print(f"   Chlorophyll stats:")
                    print(f"      Mean: {valid_chl.mean():.2f} mg/m³")
                    print(f"      Range: [{valid_chl.min():.2f}, {valid_chl.max():.2f}]")
                    print(f"      Median: {np.median(valid_chl):.2f} mg/m³")
                
                # Save output rasters
                if save_output:
                    # Extract date from filename
                    date_str = input_file.stem.split('_')[-1]  # Assumes format: tres_marias_2024-06-05
                    
                    # Save chlorophyll raster
                    chl_output = self.output_dir / f'chlorophyll_{date_str}.tif'
                    meta.update({
                        'dtype': 'float32',
                        'count': 1,
                        'nodata': -9999.0
                    })
                    
                    with rasterio.open(chl_output, 'w', **meta) as dst:
                        dst.write(chl_map, 1)
                        dst.set_band_description(1, 'Chlorophyll-a (mg/m³)')
                    
                    print(f"   ✅ Saved: {chl_output.name}")
                    
                    # Save water class raster
                    class_output = self.output_dir / f'water_class_{date_str}.tif'
                    meta.update({
                        'dtype': 'int16',
                        'nodata': -9999
                    })
                    
                    with rasterio.open(class_output, 'w', **meta) as dst:
                        dst.write(class_map, 1)
                        dst.set_band_description(1, 'Optical Water Type (1-5)')
                    
                    print(f"   ✅ Saved: {class_output.name}")
                
                return chl_map, class_map, meta
                
        except Exception as e:
            print(f"   ❌ Error processing {input_file.name}: {e}")
            return None, None, None
    
    def process_all_rasters(self):
        """
        Process all raster files in input directory
        """
        files = self.list_raster_files()
        
        if not files:
            print(f"❌ No .tif files found in {self.input_dir}")
            return
        
        print(f"\n{'='*70}")
        print(f"Processing {len(files)} raster files")
        print(f"{'='*70}")
        
        results = []
        
        for i, file in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}]", end=' ')
            
            chl_map, class_map, meta = self.process_single_raster(file, save_output=True)
            
            if chl_map is not None:
                # Extract date
                date_str = file.stem.split('_')[-1]
                
                # Calculate statistics
                valid_chl = chl_map[chl_map != -9999]
                
                if len(valid_chl) > 0:
                    results.append({
                        'date': date_str,
                        'n_pixels': len(valid_chl),
                        'chl_mean': valid_chl.mean(),
                        'chl_median': np.median(valid_chl),
                        'chl_std': valid_chl.std(),
                        'chl_min': valid_chl.min(),
                        'chl_max': valid_chl.max()
                    })
        
        # Save summary
        if results:
            df = pd.DataFrame(results)
            df.to_csv(self.output_dir / 'chlorophyll_summary.csv', index=False)
            
            print(f"\n{'='*70}")
            print(f"PROCESSING COMPLETE")
            print(f"{'='*70}")
            print(f"Processed: {len(results)} files")
            print(f"Output directory: {self.output_dir}")
            print(f"Summary saved: chlorophyll_summary.csv")
            print(f"\nOverall statistics:")
            print(f"  Mean chlorophyll: {df['chl_mean'].mean():.2f} mg/m³")
            print(f"  Range: [{df['chl_min'].min():.2f}, {df['chl_max'].max():.2f}]")
            print(f"{'='*70}\n")
            
            return df
        
        return None
    
    def create_rgb_preview(self, raster_file, output_file=None):
        """
        Create an RGB preview of a raster for visualization
        (Uses Rrs bands to create a natural color composite)
        """
        import matplotlib.pyplot as plt
        
        with rasterio.open(raster_file) as src:
            # Read RGB bands (Rrs665=Red, Rrs560=Green, Rrs490=Blue)
            # These correspond to bands 4, 3, 2
            red = src.read(4)
            green = src.read(3)
            blue = src.read(2)
            
            # Stack into RGB
            rgb = np.dstack([red, green, blue])
            
            # Normalize to 0-1 for display
            rgb_normalized = np.clip(rgb / np.percentile(rgb, 99), 0, 1)
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(rgb_normalized)
            ax.set_title(f'RGB Preview: {raster_file.name}', fontsize=14)
            ax.axis('off')
            
            if output_file:
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"Preview saved: {output_file}")
            
            plt.tight_layout()
            return fig

def main():
    """
    Main execution function
    """
    print("="*70)
    print("RASTER PROCESSOR - Apply Chl-CONNECT to Full Images")
    print("="*70)
    print("\nThis script processes GeoTIFF files exported from Google Earth Engine")
    print("and calculates chlorophyll-a for every pixel using Chl-CONNECT.\n")
    
    # Check if input directory exists
    input_dir = 'downloaded_rasters'
    if not Path(input_dir).exists():
        print(f"⚠️  Input directory '{input_dir}' not found!")
        print(f"\nPlease:")
        print(f"1. Run export_full_rasters.py to export images from GEE")
        print(f"2. Download the GeoTIFF files from Google Drive")
        print(f"3. Place them in a '{input_dir}/' folder")
        print(f"4. Run this script again\n")
        return
    
    # Initialize processor
    processor = RasterProcessor(
        input_dir=input_dir,
        output_dir='chlorophyll_rasters'
    )
    
    # Process all rasters
    summary_df = processor.process_all_rasters()
    
    if summary_df is not None:
        print("\n✅ All rasters processed successfully!")
        print(f"\nYou can now:")
        print(f"1. Open the GeoTIFF files in QGIS/ArcGIS")
        print(f"2. Create maps and visualizations")
        print(f"3. Analyze spatial patterns")
        print(f"4. Compare temporal changes\n")
    else:
        print("\n❌ Processing failed or no valid data found\n")

if __name__ == "__main__":
    main()