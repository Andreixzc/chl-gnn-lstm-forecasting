"""
Enhanced Visualization for GNN Predictions
Creates smooth, interpolated intensity maps from discrete pixel predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.interpolate import griddata, Rbf
from matplotlib.colors import LinearSegmentedColormap
import os
import warnings
warnings.filterwarnings('ignore')

# Try importing optional advanced interpolation libraries
try:
    from scipy.interpolate import SmoothBivariateSpline
    HAS_SPLINE = True
except ImportError:
    HAS_SPLINE = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF as RBF_Kernel, WhiteKernel
    HAS_GAUSSIAN_PROCESS = True
except ImportError:
    HAS_GAUSSIAN_PROCESS = False

class SmoothMapVisualizer:
    """Create smooth interpolated maps from discrete predictions"""
    
    def __init__(self, aoi_path="Area.json"):
        """
        Initialize visualizer
        
        Args:
            aoi_path: Path to Area.json for reservoir boundaries
        """
        self.aoi_path = aoi_path
        self.gdf = None
        self.load_boundaries()
        
    def load_boundaries(self):
        """Load reservoir boundary shapefile"""
        try:
            self.gdf = gpd.read_file(self.aoi_path)
            bounds = self.gdf.total_bounds
            self.minx, self.miny, self.maxx, self.maxy = bounds
            print(f"✅ Loaded reservoir boundaries from {self.aoi_path}")
        except Exception as e:
            print(f"⚠️ Could not load boundaries: {e}")
            self.gdf = None
    
    def create_smooth_interpolation(self, coords, values, grid_resolution=200, method='rbf'):
        """
        Create smooth interpolated surface from discrete points
        
        Args:
            coords: Array of [lon, lat] coordinates, shape (n_points, 2)
            values: Array of values at each coordinate, shape (n_points,)
            grid_resolution: Resolution of interpolation grid
            method: Interpolation method - 'rbf' (Radial Basis Function), 
                   'cubic' (griddata cubic), 'gaussian' (Gaussian Process),
                   'idw' (Inverse Distance Weighting), or 'thin_plate' (Thin Plate Spline)
            
        Returns:
            grid_lon, grid_lat, grid_values: Interpolated grid arrays
        """
        # Create regular grid
        grid_lon = np.linspace(self.minx, self.maxx, grid_resolution)
        grid_lat = np.linspace(self.miny, self.maxy, grid_resolution)
        grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
        
        # Choose interpolation method
        if method == 'rbf':
            # Radial Basis Function - SMOOTHEST, RECOMMENDED
            grid_values = self._interpolate_rbf(coords, values, grid_lon_mesh, grid_lat_mesh)
        
        elif method == 'gaussian' and HAS_GAUSSIAN_PROCESS:
            # Gaussian Process - Very smooth but slower
            grid_values = self._interpolate_gaussian_process(coords, values, grid_lon_mesh, grid_lat_mesh)
        
        elif method == 'idw':
            # Inverse Distance Weighting - Fast and smooth
            grid_values = self._interpolate_idw(coords, values, grid_lon_mesh, grid_lat_mesh)
        
        elif method == 'thin_plate':
            # Thin Plate Spline - RBF with 'thin_plate' function
            grid_values = self._interpolate_thin_plate(coords, values, grid_lon_mesh, grid_lat_mesh)
        
        else:
            # Default: cubic griddata (your original method)
            grid_values = griddata(
                points=coords,
                values=values,
                xi=(grid_lon_mesh, grid_lat_mesh),
                method='cubic',
                fill_value=np.nan
            )
        
        # Mask areas outside the water body
        if self.gdf is not None:
            from shapely.geometry import Point
            
            # Create mask for water areas
            water_mask = np.zeros_like(grid_values, dtype=bool)
            
            for i in range(grid_resolution):
                for j in range(grid_resolution):
                    point = Point(grid_lon_mesh[i, j], grid_lat_mesh[i, j])
                    
                    # Check if point is in any water polygon
                    for idx, row in self.gdf.iterrows():
                        if row.geometry.contains(point) or row.geometry.intersects(point):
                            water_mask[i, j] = True
                            break
            
            # Apply mask
            grid_values = np.where(water_mask, grid_values, np.nan)
        
        return grid_lon_mesh, grid_lat_mesh, grid_values
    
    def _interpolate_rbf(self, coords, values, grid_lon, grid_lat):
        """Radial Basis Function interpolation - produces very smooth surfaces"""
        print(f"      Using RBF interpolation (Multiquadric)...")
        
        # Create RBF interpolator
        rbf = Rbf(coords[:, 0], coords[:, 1], values, 
                  function='multiquadric',  # 'multiquadric', 'gaussian', 'thin_plate', 'cubic', 'quintic'
                  smooth=0.1)  # Smoothing parameter (0 = exact interpolation)
        
        # Interpolate
        grid_values = rbf(grid_lon, grid_lat)
        
        return grid_values
    
    def _interpolate_thin_plate(self, coords, values, grid_lon, grid_lat):
        """Thin Plate Spline interpolation - minimizes bending energy"""
        print(f"      Using Thin Plate Spline interpolation...")
        
        # Create RBF interpolator with thin_plate function
        rbf = Rbf(coords[:, 0], coords[:, 1], values, 
                  function='thin_plate',
                  smooth=0.05)
        
        # Interpolate
        grid_values = rbf(grid_lon, grid_lat)
        
        return grid_values
    
    def _interpolate_idw(self, coords, values, grid_lon, grid_lat, power=2):
        """Inverse Distance Weighting interpolation"""
        print(f"      Using IDW interpolation (power={power})...")
        
        # Flatten grid
        grid_points = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
        grid_values = np.zeros(len(grid_points))
        
        # Calculate IDW for each grid point
        for i, grid_point in enumerate(grid_points):
            # Calculate distances to all sample points
            distances = np.sqrt(np.sum((coords - grid_point)**2, axis=1))
            
            # Avoid division by zero
            distances = np.where(distances < 1e-10, 1e-10, distances)
            
            # Calculate weights (inverse distance with power)
            weights = 1.0 / (distances ** power)
            
            # Weighted average
            grid_values[i] = np.sum(weights * values) / np.sum(weights)
        
        # Reshape back to grid
        grid_values = grid_values.reshape(grid_lon.shape)
        
        return grid_values
    
    def _interpolate_gaussian_process(self, coords, values, grid_lon, grid_lat):
        """Gaussian Process interpolation - highest quality but slower"""
        print(f"      Using Gaussian Process interpolation...")
        
        # Create kernel
        kernel = RBF_Kernel(length_scale=0.1) + WhiteKernel(noise_level=0.01)
        
        # Create GP regressor
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        
        # Fit
        gp.fit(coords, values)
        
        # Predict
        grid_points = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
        grid_values = gp.predict(grid_points)
        
        # Reshape
        grid_values = grid_values.reshape(grid_lon.shape)
        
        return grid_values
    
    def visualize_smooth_predictions(self, predictions, coords, 
                                     save_path="smooth_predictions.png",
                                     cmap='RdYlGn_r', figsize=(18, 12),
                                     save_individual=True, output_dir="smooth_maps_individual"):
        """
        Create smooth interpolated maps for all prediction time steps
        
        Args:
            predictions: Prediction array, shape (n_pixels, n_timesteps)
            coords: Coordinate array, shape (n_pixels, 2) [lon, lat]
            save_path: Where to save the combined output
            cmap: Colormap to use
            figsize: Figure size
            save_individual: Whether to save individual maps for each time step
            output_dir: Directory to save individual maps
        """
        n_steps = predictions.shape[1]
        
        # Create output directory for individual maps if needed
        if save_individual:
            os.makedirs(output_dir, exist_ok=True)
        
        # Create subplot grid
        cols = 3
        rows = (n_steps + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Global min/max for consistent colorbar
        vmin, vmax = predictions.min(), predictions.max()
        
        print(f"🎨 Creating {n_steps} smooth interpolated maps...")
        
        for step in range(n_steps):
            row, col = step // cols, step % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Get predictions for this time step
            step_predictions = predictions[:, step]
            
            print(f"   Processing time step {step + 1}/{n_steps}...", end='\r')
            
            # Create smooth interpolation
            grid_lon, grid_lat, grid_values = self.create_smooth_interpolation(
                coords, step_predictions, grid_resolution=200
            )
            
            # Plot water boundaries
            if self.gdf is not None:
                self.gdf.plot(ax=ax, color='lightgray', alpha=0.3, edgecolor='gray', linewidth=0.5)
            
            # Plot smooth interpolated surface
            im = ax.contourf(grid_lon, grid_lat, grid_values,
                           levels=20, cmap=cmap, vmin=vmin, vmax=vmax,
                           alpha=0.8, extend='both')
            
            # Overlay original sample points (optional)
            ax.scatter(coords[:, 0], coords[:, 1], c='black', s=2, alpha=0.3, 
                      label='Sample points')
            
            ax.set_xlabel('Longitude', fontsize=10)
            ax.set_ylabel('Latitude', fontsize=10)
            ax.set_title(f'Future Time Step {step + 1}', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            
            # Add colorbar for each subplot
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Chlorophyll-a (mg/m³)', fontsize=9)
            
            # Save individual map if requested
            if save_individual:
                self._save_individual_smooth_map(
                    grid_lon, grid_lat, grid_values, coords, step,
                    vmin, vmax, cmap, output_dir
                )
        
        print(f"   Processing time step {n_steps}/{n_steps}... Done!")
        
        # Hide empty subplots
        for step in range(n_steps, rows * cols):
            if rows > 1:
                row, col = step // cols, step % cols
                axes[row, col].axis('off')
            else:
                if step < len(axes):
                    axes[step].axis('off')
        
        plt.suptitle('GNN Predictions - Smooth Interpolated Chlorophyll Maps\nTrês Marias Reservoir', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Smooth maps saved: {save_path}")
    
    def visualize_single_smooth_map(self, values, coords, title="Chlorophyll-a",
                                    save_path="single_smooth_map.png",
                                    cmap='RdYlGn_r', figsize=(12, 10)):
        """
        Create a single high-quality smooth map
        
        Args:
            values: Array of values at each coordinate, shape (n_points,)
            coords: Coordinate array, shape (n_pixels, 2) [lon, lat]
            title: Map title
            save_path: Where to save the output
            cmap: Colormap to use
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        print(f"🎨 Creating single smooth interpolated map...")
        
        # Create smooth interpolation with higher resolution
        grid_lon, grid_lat, grid_values = self.create_smooth_interpolation(
            coords, values, grid_resolution=300
        )
        
        # Plot water boundaries
        if self.gdf is not None:
            self.gdf.plot(ax=ax, color='lightgray', alpha=0.2, edgecolor='darkgray', linewidth=1)
        
        # Plot smooth interpolated surface with contours
        levels = np.linspace(np.nanmin(grid_values), np.nanmax(grid_values), 25)
        im = ax.contourf(grid_lon, grid_lat, grid_values,
                       levels=levels, cmap=cmap, alpha=0.9, extend='both')
        
        # Add contour lines
        contours = ax.contour(grid_lon, grid_lat, grid_values,
                            levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
        
        # Overlay sample points
        ax.scatter(coords[:, 0], coords[:, 1], c='black', s=5, alpha=0.4, 
                  label=f'Sample points (n={len(coords)})', zorder=5)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Chlorophyll-a (mg/m³)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Single smooth map saved: {save_path}")
    
    def compare_discrete_vs_smooth(self, values, coords,
                                   save_path="discrete_vs_smooth.png"):
        """
        Create side-by-side comparison of discrete points vs smooth interpolation
        
        Args:
            values: Array of values, shape (n_points,)
            coords: Coordinate array, shape (n_pixels, 2)
            save_path: Where to save the comparison
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        vmin, vmax = values.min(), values.max()
        
        # Left: Discrete points
        if self.gdf is not None:
            self.gdf.plot(ax=ax1, color='lightgray', alpha=0.3, edgecolor='gray')
        
        scatter = ax1.scatter(coords[:, 0], coords[:, 1], c=values,
                            cmap='RdYlGn_r', s=80, alpha=0.8, 
                            vmin=vmin, vmax=vmax, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Discrete Sample Points\n(GNN Predictions)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Chlorophyll-a (mg/m³)')
        
        # Right: Smooth interpolation
        grid_lon, grid_lat, grid_values = self.create_smooth_interpolation(
            coords, values, grid_resolution=200
        )
        
        if self.gdf is not None:
            self.gdf.plot(ax=ax2, color='lightgray', alpha=0.2, edgecolor='gray', linewidth=0.5)
        
        im = ax2.contourf(grid_lon, grid_lat, grid_values,
                        levels=20, cmap='RdYlGn_r', vmin=vmin, vmax=vmax,
                        alpha=0.9, extend='both')
        
        # Add sample points overlay
        ax2.scatter(coords[:, 0], coords[:, 1], c='black', s=8, alpha=0.3)
        
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('Smooth Interpolated Surface\n(Cubic Interpolation)', fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        plt.colorbar(im, ax=ax2, label='Chlorophyll-a (mg/m³)')
        
        plt.suptitle('Comparison: Discrete vs Smooth Visualization', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comparison saved: {save_path}")
    
    def _save_individual_smooth_map(self, grid_lon, grid_lat, grid_values, coords, 
                                    step, vmin, vmax, cmap, output_dir):
        """Save individual smooth map for a single time step"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot water boundaries
        if self.gdf is not None:
            self.gdf.plot(ax=ax, color='lightgray', alpha=0.3, edgecolor='gray', linewidth=0.5)
        
        # Plot smooth interpolated surface
        im = ax.contourf(grid_lon, grid_lat, grid_values,
                       levels=20, cmap=cmap, vmin=vmin, vmax=vmax,
                       alpha=0.8, extend='both')
        
        # Overlay original sample points
        ax.scatter(coords[:, 0], coords[:, 1], c='black', s=2, alpha=0.3)
        
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_title(f'Smooth Interpolated Map - Time Step {step + 1}\nTrês Marias Reservoir', 
                    fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Chlorophyll-a (mg/m³)', fontsize=10)
        
        plt.tight_layout()
        individual_path = os.path.join(output_dir, f'smooth_map_step_{step+1}.png')
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_on_satellite(self, values, coords, step=1,
                              save_path="satellite_overlay.png",
                              cmap='RdYlGn_r', figsize=(14, 12),
                              zoom=13, alpha=0.6):
        """
        Create smooth interpolated map overlaid on satellite imagery
        
        Args:
            values: Array of values at each coordinate, shape (n_points,)
            coords: Coordinate array, shape (n_pixels, 2) [lon, lat]
            step: Time step number (for labeling)
            save_path: Where to save the output
            cmap: Colormap to use
            figsize: Figure size
            zoom: Satellite image zoom level (higher = more detail)
            alpha: Transparency of the overlay (0-1)
        """
        try:
            import contextily as ctx
            
            print(f"🛰️ Creating satellite overlay map...")
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create smooth interpolation
            grid_lon, grid_lat, grid_values = self.create_smooth_interpolation(
                coords, values, grid_resolution=200
            )
            
            # Convert to Web Mercator projection (required for contextily)
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            
            # Transform grid
            grid_lon_merc, grid_lat_merc = transformer.transform(grid_lon, grid_lat)
            
            # Transform sample points
            coords_merc = np.array([transformer.transform(lon, lat) for lon, lat in coords])
            
            # Plot smooth interpolated surface
            vmin, vmax = np.nanmin(grid_values), np.nanmax(grid_values)
            im = ax.contourf(grid_lon_merc, grid_lat_merc, grid_values,
                           levels=20, cmap=cmap, vmin=vmin, vmax=vmax,
                           alpha=alpha, extend='both')
            
            # Add sample points
            ax.scatter(coords_merc[:, 0], coords_merc[:, 1], 
                      c='black', s=3, alpha=0.4, zorder=5)
            
            # Add satellite basemap
            try:
                ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=zoom)
            except:
                # Fallback to OpenStreetMap if Esri fails
                ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom)
            
            ax.set_xlabel('Longitude (Web Mercator)', fontsize=11)
            ax.set_ylabel('Latitude (Web Mercator)', fontsize=11)
            ax.set_title(f'Chlorophyll-a Prediction on Satellite Imagery\nTime Step {step} - Três Marias Reservoir', 
                        fontweight='bold', fontsize=13)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Chlorophyll-a (mg/m³)', fontsize=10)
            
            # Add scalebar
            from matplotlib_scalebar.scalebar import ScaleBar
            ax.add_artist(ScaleBar(1, location='lower right'))
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Satellite overlay saved: {save_path}")
            return True
            
        except ImportError as e:
            print(f"⚠️ Satellite overlay requires contextily and matplotlib-scalebar packages")
            print(f"   Install with: pip install contextily matplotlib-scalebar")
            return False
        except Exception as e:
            print(f"⚠️ Could not create satellite overlay: {e}")
            return False
    
    def visualize_all_on_satellite(self, predictions, coords,
                                   output_dir="satellite_overlays",
                                   cmap='RdYlGn_r', zoom=13, alpha=0.6):
        """
        Create satellite overlay maps for all prediction time steps
        
        Args:
            predictions: Prediction array, shape (n_pixels, n_timesteps)
            coords: Coordinate array, shape (n_pixels, 2) [lon, lat]
            output_dir: Directory to save satellite overlays
            cmap: Colormap to use
            zoom: Satellite image zoom level
            alpha: Transparency of the overlay
        """
        os.makedirs(output_dir, exist_ok=True)
        
        n_steps = predictions.shape[1]
        print(f"🛰️ Creating {n_steps} satellite overlay maps...")
        
        success_count = 0
        for step in range(n_steps):
            step_predictions = predictions[:, step]
            save_path = os.path.join(output_dir, f'satellite_step_{step+1}.png')
            
            print(f"   Processing satellite overlay {step + 1}/{n_steps}...", end='\r')
            
            success = self.visualize_on_satellite(
                step_predictions, coords, 
                step=step+1,
                save_path=save_path,
                cmap=cmap,
                zoom=zoom,
                alpha=alpha
            )
            
            if success:
                success_count += 1
        
        print(f"   Processing satellite overlay {n_steps}/{n_steps}... Done!")
        
        if success_count > 0:
            print(f"✅ Created {success_count} satellite overlay maps in '{output_dir}/'")
        
        return success_count > 0


def add_smooth_visualization_to_forecaster(forecaster_instance):
    """
    Add smooth visualization methods to an existing GraphChlorophyllForecaster instance
    
    Args:
        forecaster_instance: An instance of GraphChlorophyllForecaster
    """
    # Create visualizer
    visualizer = SmoothMapVisualizer(aoi_path=forecaster_instance.aoi_path)
    
    def visualize_smooth_future_maps(save_path="smooth_future_maps.png"):
        """Generate smooth interpolated future prediction maps"""
        print(f"🔮 Generating smooth interpolated future maps...")
        
        if forecaster_instance.model is None:
            raise ValueError("Model not trained yet!")
        
        forecaster_instance.model.eval()
        
        # Use the last sequence from validation data
        last_data = forecaster_instance.val_data[-1].to(forecaster_instance.device)
        
        import torch
        with torch.no_grad():
            predictions = forecaster_instance.model(last_data)
        
        # Convert back to original scale
        predictions_np = predictions.cpu().numpy()
        
        # Denormalize
        original_shape = predictions_np.shape
        flat_predictions = predictions_np.reshape(-1, 1)
        denorm_predictions = forecaster_instance.dataset.scaler.inverse_transform(flat_predictions)
        final_predictions = denorm_predictions.reshape(original_shape)
        
        # Get coordinates
        coords = forecaster_instance.dataset.pixel_coords[['lon', 'lat']].values
        
        # Create smooth visualizations with individual maps
        visualizer.visualize_smooth_predictions(
            final_predictions, coords, save_path=save_path,
            save_individual=True, output_dir="smooth_maps_individual"
        )
        
        # Also create a comparison for the first time step
        visualizer.compare_discrete_vs_smooth(
            final_predictions[:, 0], coords,
            save_path="discrete_vs_smooth_comparison.png"
        )
        
        # Try to create satellite overlays for all time steps
        print("\n🛰️ Attempting to create satellite overlay maps...")
        satellite_success = visualizer.visualize_all_on_satellite(
            final_predictions, coords,
            output_dir="satellite_overlays",
            zoom=13, alpha=0.6
        )
        
        if satellite_success:
            print("✅ Satellite overlay maps created!")
        else:
            print("⚠️ Satellite overlays not available (install contextily and matplotlib-scalebar)")
        
        print(f"✅ Smooth interpolated maps created!")
        
        return final_predictions, coords
    
    # Add method to forecaster instance
    forecaster_instance.visualize_smooth_future_maps = visualize_smooth_future_maps
    
    print("✅ Added smooth visualization methods to forecaster")
    return forecaster_instance


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("SMOOTH INTERPOLATED MAP VISUALIZATION")
    print("="*60)
    print()
    print("This module adds smooth interpolation visualization to GNN predictions")
    print()
    print("Usage:")
    print("  1. Train your GNN model using GraphChlorophyllForecaster")
    print("  2. Import this module and use:")
    print("     from smooth_interpolated_visualization import add_smooth_visualization_to_forecaster")
    print("     forecaster = add_smooth_visualization_to_forecaster(forecaster)")
    print("     forecaster.visualize_smooth_future_maps()")
    print()
    print("This will create:")
    print("  • smooth_future_maps.png - All time steps with smooth interpolation")
    print("  • discrete_vs_smooth_comparison.png - Side-by-side comparison")
    print("="*60)