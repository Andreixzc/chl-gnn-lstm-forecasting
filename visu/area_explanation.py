"""
Visualize exactly what's happening with area processing
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

def visualize_area_processing():
    """Show the exact area processing approach"""
    
    print("🗺️ Visualizing Area Processing Strategy")
    print("=" * 50)
    
    # Load Area.json
    gdf = gpd.read_file('Area.json')
    bounds = gdf.total_bounds
    minx, miny, maxx, maxy = bounds
    
    print(f"📁 Area.json contains:")
    print(f"   • {len(gdf)} water features (complex polygons)")
    print(f"   • Total water area: ~768 km²")
    print()
    
    print(f"📦 Bounding Box (for Earth Engine):")
    print(f"   • Rectangle: {minx:.4f}, {miny:.4f} to {maxx:.4f}, {maxy:.4f}")
    print(f"   • Size: ~{(maxx-minx)*111:.0f} × {(maxy-miny)*111:.0f} km")
    print(f"   • Area: ~{(maxx-minx)*111 * (maxy-miny)*111:.0f} km²")
    print(f"   • Water coverage: {768/((maxx-minx)*111 * (maxy-miny)*111)*100:.1f}%")
    print()
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Area.json water shapes
    gdf.plot(ax=ax1, color='blue', alpha=0.7, edgecolor='darkblue')
    ax1.set_title('1. Area.json\n(Exact Water Shapes)', fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)
    
    # 2. Bounding box for Earth Engine
    gdf.plot(ax=ax2, color='lightblue', alpha=0.5, edgecolor='blue')
    # Draw bounding box
    bbox_x = [minx, maxx, maxx, minx, minx]
    bbox_y = [miny, miny, maxy, maxy, miny]
    ax2.plot(bbox_x, bbox_y, 'r-', linewidth=3, label='Earth Engine Query Box')
    ax2.fill(bbox_x, bbox_y, 'red', alpha=0.1)
    ax2.set_title('2. Earth Engine Query\n(Simple Bounding Box)', fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Grid point generation
    gdf.plot(ax=ax3, color='lightblue', alpha=0.5, edgecolor='blue')
    
    # Create a sample grid (smaller for visualization)
    grid_size = 15  # Small grid for visualization
    x_step = (maxx - minx) / grid_size
    y_step = (maxy - miny) / grid_size
    
    water_points = []
    land_points = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = minx + (i + 0.5) * x_step
            y = miny + (j + 0.5) * y_step
            point = Point(x, y)
            
            # Test if point is in water
            is_water = False
            for idx, row in gdf.iterrows():
                if row.geometry.contains(point) or row.geometry.intersects(point):
                    is_water = True
                    break
            
            if is_water:
                water_points.append((x, y))
            else:
                land_points.append((x, y))
    
    # Plot grid points
    if land_points:
        land_x, land_y = zip(*land_points)
        ax3.scatter(land_x, land_y, c='red', s=20, alpha=0.7, label=f'Land points ({len(land_points)})')
    
    if water_points:
        water_x, water_y = zip(*water_points)
        ax3.scatter(water_x, water_y, c='green', s=20, alpha=0.9, label=f'Water points ({len(water_points)})')
    
    ax3.set_title('3. Grid Point Testing\n(Keep Only Water Points)', fontweight='bold')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Final result
    gdf.plot(ax=ax4, color='lightblue', alpha=0.5, edgecolor='blue')
    if water_points:
        ax4.scatter(water_x, water_y, c='green', s=30, alpha=0.9, 
                   label=f'Final Sampling Points ({len(water_points)})')
    ax4.set_title('4. Final Result\n(Water-Only Sampling)', fontweight='bold')
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('area_processing_explanation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved as 'area_processing_explanation.png'")
    print()
    
    print(f"🎯 SUMMARY:")
    print(f"   • Area.json = Exact water shapes (complex)")
    print(f"   • Earth Engine = Simple bounding box (fast queries)")
    print(f"   • Pixel selection = Test each point against exact shapes")
    print(f"   • Result = Only water pixels kept, land pixels discarded")
    print()
    
    print(f"📊 In your actual run:")
    print(f"   • Tested: 5,929 grid points in bounding box")
    print(f"   • Kept: 717 points (inside water shapes)")
    print(f"   • Discarded: 5,212 points (on land)")
    print(f"   • Success rate: 12.1%")

if __name__ == "__main__":
    visualize_area_processing()
