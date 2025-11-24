"""
Full Raster Export for Três Marias Reservoir
Exports complete satellite images (all pixels) as GeoTIFF files
One file per date with automatic tile mosaicking

Author: Water Quality Remote Sensing
"""

import ee
import json
from datetime import datetime, timedelta
import math

# Initialize Earth Engine
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()

class RasterExporter:
    def __init__(self, aoi, start_date='2024-06-01', end_date='2024-11-30'):
        """
        Initialize the raster exporter
        
        Args:
            aoi: ee.Geometry of the area of interest
            start_date: Start date for export
            end_date: End date for export
        """
        self.aoi = aoi
        self.start_date = start_date
        self.end_date = end_date
        
    def mask_clouds_and_water(self, image):
        """
        Apply cloud and water masks to Sentinel-2 image
        """
        # Cloud mask using QA60
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        cloud_mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
                    qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        
        # Water mask using NDWI
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        # Basic water mask (less restrictive)
        water_mask = ndwi.gt(-0.1).And(
                    image.select('B8').lt(2000)).And(
                    image.select('B4').lt(2000))
        
        # Combine masks
        final_mask = cloud_mask.And(water_mask)
        
        return image.updateMask(final_mask)
    
    def calculate_rrs(self, image):
        """
        Convert Sentinel-2 surface reflectance to Remote Sensing Reflectance (Rrs)
        """
        # Sentinel-2 bands for water quality (scaled 0-10000)
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A']
        
        # Convert to reflectance (0-1)
        reflectance = image.select(bands).multiply(0.0001)
        
        # Convert to Rrs (reflectance / π)
        rrs = reflectance.divide(math.pi)
        
        # Rename bands to match CHL-CONNECT requirements for MSI
        rrs_bands = rrs.select(bands, 
                              ['Rrs443', 'Rrs490', 'Rrs560', 'Rrs665', 
                               'Rrs705', 'Rrs740', 'Rrs783', 'Rrs865'])
        
        return image.addBands(rrs_bands)
    
    def get_daily_collections(self):
        """
        Get collection grouped by date
        Returns a list of (date_string, image_collection) tuples
        """
        print("Loading Sentinel-2 collection...")
        
        # Load full collection
        s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(self.aoi)
                        .filterDate(self.start_date, self.end_date)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))
        
        collection_size = s2_collection.size().getInfo()
        print(f"Found {collection_size} images in date range")
        
        if collection_size == 0:
            print("No valid images found!")
            return []
        
        # Get all unique dates
        dates = s2_collection.aggregate_array('system:time_start').getInfo()
        unique_dates = sorted(set([datetime.fromtimestamp(d/1000).strftime('%Y-%m-%d') 
                                   for d in dates]))
        
        print(f"Found {len(unique_dates)} unique dates")
        
        daily_collections = []
        for date_str in unique_dates:
            # Filter to this specific date
            daily = (s2_collection
                    .filterDate(date_str, 
                               (datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=1))
                               .strftime('%Y-%m-%d')))
            
            n_images = daily.size().getInfo()
            daily_collections.append((date_str, daily, n_images))
        
        return daily_collections
    
    def export_daily_rasters(self, output_folder='tres_marias_rasters', 
                            drive_folder='GEE_Exports'):
        """
        Export one raster per date to Google Drive
        
        Args:
            output_folder: Prefix for export filenames
            drive_folder: Google Drive folder name
        """
        daily_collections = self.get_daily_collections()
        
        if not daily_collections:
            print("No data to export!")
            return
        
        print(f"\n{'='*70}")
        print(f"Starting export of {len(daily_collections)} daily rasters")
        print(f"{'='*70}\n")
        
        tasks = []
        
        for date_str, daily_collection, n_tiles in daily_collections:
            print(f"📅 {date_str}:")
            print(f"   {n_tiles} tile(s) covering area")
            
            # Apply processing
            processed = (daily_collection
                        .map(self.mask_clouds_and_water)
                        .map(self.calculate_rrs))
            
            # Mosaic if multiple tiles exist
            if n_tiles > 1:
                print(f"   Mosaicking {n_tiles} tiles...")
                # Get tile IDs
                tiles = daily_collection.aggregate_array('MGRS_TILE').getInfo()
                print(f"   Tiles: {tiles}")
                
            # Create mosaic (combines all tiles for this date)
            mosaicked = processed.mosaic()
            
            # Select Rrs bands for export
            export_bands = ['Rrs443', 'Rrs490', 'Rrs560', 'Rrs665', 'Rrs705', 
                           'Rrs740', 'Rrs783', 'Rrs865']
            
            # Create export task
            task = ee.batch.Export.image.toDrive(
                image=mosaicked.select(export_bands).clip(self.aoi),
                description=f'{output_folder}_{date_str}',
                folder=drive_folder,
                fileNamePrefix=f'{output_folder}_{date_str}',
                scale=20,  # 20m resolution for Sentinel-2
                region=self.aoi.getInfo()['coordinates'],
                fileFormat='GeoTIFF',
                maxPixels=1e10,  # Increase for large areas
                crs='EPSG:4326'
            )
            
            # Start the task
            task.start()
            tasks.append({
                'date': date_str,
                'task_id': task.id,
                'status': task.status()
            })
            
            print(f"   ✅ Export task started: {task.id}")
            print(f"   Status: {task.status()['state']}\n")
        
        # Print summary
        print(f"{'='*70}")
        print(f"EXPORT SUMMARY")
        print(f"{'='*70}")
        print(f"Total exports initiated: {len(tasks)}")
        print(f"Google Drive folder: {drive_folder}")
        print(f"\nTo check status, visit:")
        print(f"https://code.earthengine.google.com/tasks")
        print(f"\nExports will appear in your Google Drive as:")
        for task in tasks[:3]:  # Show first 3 as examples
            print(f"  - {output_folder}_{task['date']}.tif")
        if len(tasks) > 3:
            print(f"  ... and {len(tasks)-3} more")
        print(f"{'='*70}\n")
        
        return tasks
    
    def check_export_status(self):
        """
        Check status of all running export tasks
        """
        tasks = ee.batch.Task.list()
        
        print(f"\n{'='*70}")
        print(f"EXPORT TASK STATUS")
        print(f"{'='*70}\n")
        
        running = 0
        completed = 0
        failed = 0
        
        for task in tasks[:20]:  # Show last 20 tasks
            state = task.status()['state']
            task_type = task.status()['description']
            
            if state == 'RUNNING':
                running += 1
                status_icon = '⏳'
            elif state == 'COMPLETED':
                completed += 1
                status_icon = '✅'
            elif state == 'FAILED':
                failed += 1
                status_icon = '❌'
            else:
                status_icon = '⏸️'
            
            print(f"{status_icon} {state:12s} | {task_type}")
        
        print(f"\n{'='*70}")
        print(f"Summary: {running} running, {completed} completed, {failed} failed")
        print(f"{'='*70}\n")

def main():
    """
    Main execution function
    """
    # Load AOI from Area.json
    try:
        with open('Area.json', 'r') as f:
            aoi_data = json.load(f)
        coords = aoi_data['features'][0]['geometry']['coordinates']
        aoi = ee.Geometry.Polygon(coords)
        print("✅ Loaded AOI from Area.json")
    except Exception as e:
        print(f"⚠️ Error loading Area.json: {e}")
        # Fallback coordinates for Três Marias
        aoi = ee.Geometry.Polygon([[
            [-45.559114, -18.954365], 
            [-45.559114, -18.212409], 
            [-44.839706, -18.212409], 
            [-44.839706, -18.954365], 
            [-45.559114, -18.954365]
        ]])
        print("✅ Using fallback AOI coordinates")
    
    # Calculate AOI area
    area_km2 = aoi.area().divide(1000000).getInfo()
    print(f"AOI area: {area_km2:.2f} km²")
    
    # Estimate pixels
    pixel_size = 20  # meters
    estimated_pixels = area_km2 * 1e6 / (pixel_size ** 2)
    print(f"Estimated pixels per image: {estimated_pixels:,.0f}")
    print(f"At 20m resolution\n")
    
    # Initialize exporter
    exporter = RasterExporter(
        aoi=aoi,
        start_date='2024-06-01',
        end_date='2024-11-30'
    )
    
    print("="*70)
    print("FULL RASTER EXPORT - Três Marias Reservoir")
    print("="*70)
    print("This will export complete satellite images (all pixels)")
    print("Each date will be saved as a separate GeoTIFF file")
    print("Files will be saved to your Google Drive")
    print("="*70 + "\n")
    
    # Start exports
    tasks = exporter.export_daily_rasters(
        output_folder='tres_marias',
        drive_folder='Tres_Marias_Rasters'
    )
    
    print("\n⚠️  IMPORTANT NOTES:")
    print("1. Exports are asynchronous - they run in the background")
    print("2. Check progress at: https://code.earthengine.google.com/tasks")
    print("3. Files will appear in Google Drive when complete")
    print("4. Large exports may take 10-30 minutes per image")
    print("5. Run exporter.check_export_status() to monitor progress\n")
    
    return exporter

if __name__ == "__main__":
    exporter = main()
    
    # Uncomment to check status after running:
    # import time
    # time.sleep(60)  # Wait 1 minute
    # exporter.check_export_status()