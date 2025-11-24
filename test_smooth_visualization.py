import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
import contextily as ctx
import os
import glob
import warnings

# Optional: Try to import geemap (it might not be installed in standard python envs)
try:
    import geemap
    import ee
    HAS_GEEMAP = True
except ImportError:
    HAS_GEEMAP = False

warnings.filterwarnings('ignore')

class CompleteMapGenerator:
    def __init__(self, aoi_path="Area.json"):
        self.aoi_path = aoi_path
        self.gdf = None
        self.df = None
        self.grid_img = None
        self.transform = None
        self.bounds = None
        self.crs = "EPSG:4326" # Standard Lat/Lon
        
        # Load Data
        self.load_boundaries()
        self.load_latest_prediction()
        
    def load_boundaries(self):
        """Load vector boundaries"""
        if os.path.exists(self.aoi_path):
            self.gdf = gpd.read_file(self.aoi_path)
            self.bounds = self.gdf.total_bounds # [minx, miny, maxx, maxy]
            print(f"✅ Loaded Boundaries: {self.aoi_path}")
        else:
            raise FileNotFoundError(f"❌ Could not find {self.aoi_path}")

    def load_latest_prediction(self):
        """Find and load the first prediction CSV"""
        files = sorted(glob.glob("predictions_csv/prediction_step_*.csv"))
        if not files:
            raise FileNotFoundError("❌ No prediction CSV files found!")
        
        print(f"📂 Loading Data: {files[0]}")
        self.df = pd.read_csv(files[0])
        self.points = self.df[['lon', 'lat']].values
        self.values = self.df['chlorophyll_a_predicted'].values

    def generate_smooth_surface(self, width=1500):
        """Generate the high-res, organic smooth surface"""
        print(f"🔄 Generating High-Res Surface ({width}px wide)...")
        
        minx, miny, maxx, maxy = self.bounds
        pixel_size = (maxx - minx) / width
        
        # Calculate Height
        height = int(np.ceil((maxy - miny) / pixel_size))
        actual_miny = maxy - (height * pixel_size)
        
        # Save dimensions for later
        self.shape = (height, width)
        self.extent = [minx, maxx, actual_miny, maxy]
        
        # Generate Grid Centers
        grid_x = np.linspace(minx + pixel_size/2, maxx - pixel_size/2, width)
        grid_y = np.linspace(actual_miny + pixel_size/2, maxy - pixel_size/2, height)
        grid_x_mesh, grid_y_mesh = np.meshgrid(grid_x, grid_y)
        
        # 1. Cubic Interpolation (The Structure)
        grid_cubic = griddata(self.points, self.values, (grid_x_mesh, grid_y_mesh), method='cubic')
        
        # 2. Nearest Neighbor (The Filler)
        grid_near = griddata(self.points, self.values, (grid_x_mesh, grid_y_mesh), method='nearest')
        
        # 3. Combine
        grid_final = np.where(np.isnan(grid_cubic), grid_near, grid_cubic)
        
        # 4. Flip & Blur (The Organic Fluid Look)
        grid_img = np.flipud(grid_final)
        self.grid_img = gaussian_filter(grid_img, sigma=2)
        
        # 5. Apply Water Mask
        self.transform = from_origin(minx, maxy, pixel_size, pixel_size)
        
        mask = rasterize(
            [(geom, 1) for geom in self.gdf.geometry],
            out_shape=self.shape,
            transform=self.transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )
        
        # Set land to NaN
        self.grid_img[mask == 0] = np.nan
        print("✨ Smooth surface generated.")

    def export_geotiff(self, filename="smooth_map_final.tif"):
        """Export GeoTIFF (Required for Geemap)"""
        print(f"💾 Exporting GeoTIFF: {filename}")
        
        # Convert NaNs to a nodata value for the file
        data_to_write = np.nan_to_num(self.grid_img, nan=-9999)
        
        with rasterio.open(
            filename, 'w',
            driver='GTiff',
            height=self.shape[0],
            width=self.shape[1],
            count=1,
            dtype='float32',
            crs=self.crs,
            transform=self.transform,
            nodata=-9999
        ) as dst:
            dst.write(data_to_write.astype('float32'), 1)
            
        return filename

    def save_static_satellite_map(self, output_path="static_satellite_map.png"):
        """Generate a static PNG with Satellite Background"""
        print("🛰️ Generating Static Satellite Map...")
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # 1. Plot the Smooth Raster
        # We use alpha=0.7 so we can see the water texture underneath
        im = ax.imshow(self.grid_img, 
                       extent=self.extent, 
                       origin='upper', 
                       cmap='RdYlGn_r', 
                       alpha=0.7,
                       zorder=2) # On top of map
        
        # 2. Add Vector Outline (for precision check)
        self.gdf.plot(ax=ax, facecolor='none', edgecolor='black', 
                     linewidth=0.5, alpha=0.5, zorder=3)
        
        # 3. Add Satellite Basemap using Contextily
        # crs="EPSG:4326" tells contextily our plot is in Lat/Lon
        try:
            ctx.add_basemap(ax, 
                           crs=self.crs, 
                           source=ctx.providers.Esri.WorldImagery,
                           zoom=13)
            print("   ✓ Added Esri World Imagery")
        except Exception as e:
            print(f"   ⚠️ Could not fetch tiles: {e}")
        
        # Formatting
        ax.set_title("Predicted Chlorophyll-a Intensity", fontweight='bold', fontsize=14)
        ax.axis('off') # Clean look
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('mg/m³')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved Image: {output_path}")

    def launch_interactive_geemap(self, geotiff_path):
        """
        Launch interactive map if inside a Jupyter environment.
        If running as script, it prints the code you need.
        """
        print("\n🌍 PREPARING INTERACTIVE MAP...")
        
        # Calculate min/max for visualization
        valid_data = self.grid_img[~np.isnan(self.grid_img)]
        vmin, vmax = float(np.min(valid_data)), float(np.max(valid_data))
        
        # The code snippet to run in Jupyter
        jupyter_code = f"""
# COPY THIS INTO A JUPYTER CELL TO VIEW MAP:
import geemap
import os

# 1. Initialize Map
Map = geemap.Map()
Map.add_basemap('SATELLITE')

# 2. Define Path to the GeoTIFF we just made
geotiff_path = "{geotiff_path}"

# 3. Add the Layer
Map.add_raster(
    geotiff_path, 
    layer_name='Chlorophyll Smooth',
    colormap='RdYlGn_r',  # Reverse Red-Yellow-Green
    vmin={vmin:.2f}, 
    vmax={vmax:.2f},
    nodata=-9999,
    opacity=0.7
)

# 4. Center and Show
Map.centerObject(geemap.raster_bounds(geotiff_path), 13)
Map
        """
        
        if HAS_GEEMAP:
            print("   (Geemap detected - Attempting to render object...)")
            # Note: This only shows if you are running THIS script in Jupyter
            try:
                Map = geemap.Map()
                Map.add_basemap('SATELLITE')
                Map.add_raster(
                    geotiff_path, 
                    layer_name='Chlorophyll Smooth', 
                    palette=['green', 'yellow', 'red'], # Simple palette for example
                    vmin=vmin, vmax=vmax, 
                    nodata=-9999, opacity=0.7
                )
                Map.centerObject(geemap.raster_bounds(geotiff_path), 13)
                return Map
            except Exception as e:
                print(f"   Could not auto-launch geemap: {e}")
                print("   Please use the code snippet below.")
        
        print("-" * 60)
        print(jupyter_code)
        print("-" * 60)

# --- Execution ---
if __name__ == "__main__":
    # 1. Init
    generator = CompleteMapGenerator()
    
    # 2. Generate Data
    generator.generate_smooth_surface(width=1500)
    
    # 3. Export GeoTIFF (Crucial for Geemap)
    tif_path = generator.export_geotiff("chlorophyll_smooth.tif")
    
    # 4. Generate Static Satellite PNG
    generator.save_static_satellite_map("satellite_overlay_hd.png")
    
    # 5. Interactive Map instructions
    generator.launch_interactive_geemap(tif_path)