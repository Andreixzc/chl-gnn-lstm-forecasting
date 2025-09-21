"""
Graph-Based Spatial Features for Chlorophyll Prediction
Creates adjacency graph of water pixels for realistic spatial modeling
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class WaterPixelGraph:
    """Build adjacency graph of water pixels"""
    
    def __init__(self, timeseries_data):
        self.data = timeseries_data
        self.pixel_coordinates = None
        self.adjacency_graph = defaultdict(list)
        self.coordinate_to_id = {}
        
    def get_unique_pixels(self):
        """Get unique water pixel coordinates"""
        self.pixel_coordinates = self.data[['lon', 'lat']].drop_duplicates().values
        
        # Create coordinate lookup
        for i, coord in enumerate(self.pixel_coordinates):
            coord_key = f"{coord[0]:.6f}_{coord[1]:.6f}"
            self.coordinate_to_id[coord_key] = i
            
        print(f"📍 Found {len(self.pixel_coordinates)} unique water pixels")
        return self.pixel_coordinates
    
    def build_adjacency_graph(self, max_distance=0.025):
        """
        Build adjacency graph based on actual proximity
        Only connects pixels that are truly adjacent in water
        max_distance: threshold for considering pixels as neighbors (degrees)
        """
        print("🔗 Building water pixel adjacency graph...")
        
        if self.pixel_coordinates is None:
            self.get_unique_pixels()
        
        # For each pixel, find its true neighbors
        for i, coord_i in enumerate(self.pixel_coordinates):
            neighbors = []
            
            for j, coord_j in enumerate(self.pixel_coordinates):
                if i != j:  # Don't include self
                    # Calculate distance
                    distance = np.sqrt((coord_i[0] - coord_j[0])**2 + (coord_i[1] - coord_j[1])**2)
                    
                    # Only consider as neighbor if within threshold
                    if distance <= max_distance:
                        neighbors.append(j)
            
            coord_key = f"{coord_i[0]:.6f}_{coord_i[1]:.6f}"
            self.adjacency_graph[coord_key] = neighbors
        
        # Print connectivity stats
        neighbor_counts = [len(neighbors) for neighbors in self.adjacency_graph.values()]
        print(f"✅ Graph built - Average neighbors per pixel: {np.mean(neighbor_counts):.1f}")
        print(f"   Min neighbors: {np.min(neighbor_counts)}, Max neighbors: {np.max(neighbor_counts)}")
        
        return self.adjacency_graph
    
    def visualize_graph(self, sample_size=50):
        """Visualize a sample of the adjacency graph"""
        if not self.adjacency_graph:
            self.build_adjacency_graph()
        
        plt.figure(figsize=(12, 8))
        
        # Plot all pixels
        coords = self.pixel_coordinates
        plt.scatter(coords[:, 0], coords[:, 1], c='lightblue', s=30, alpha=0.6, label='Water pixels')
        
        # Sample some connections to visualize
        sample_coords = coords[:sample_size] if len(coords) > sample_size else coords
        
        for coord in sample_coords:
            coord_key = f"{coord[0]:.6f}_{coord[1]:.6f}"
            if coord_key in self.adjacency_graph:
                neighbor_indices = self.adjacency_graph[coord_key]
                
                for neighbor_idx in neighbor_indices:
                    neighbor_coord = coords[neighbor_idx]
                    plt.plot([coord[0], neighbor_coord[0]], [coord[1], neighbor_coord[1]], 
                            'k-', alpha=0.3, linewidth=0.5)
        
        plt.scatter(sample_coords[:, 0], sample_coords[:, 1], c='red', s=50, 
                   label=f'Sample pixels with connections ({len(sample_coords)})')
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Water Pixel Adjacency Graph (Sample)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('water_pixel_graph.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Graph visualization saved: water_pixel_graph.png")

class GraphBasedChlorophyllForecaster:
    """Forecasting with graph-based spatial features"""
    
    def __init__(self, data_path="chl_connect_timeseries_2000pts.csv"):
        self.data_path = data_path
        self.data = None
        self.graph = None
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the time series data"""
        print("📊 Loading chlorophyll time series data...")
        self.data = pd.read_csv(self.data_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        print(f"✅ Loaded {len(self.data)} observations")
        return self.data
    
    def build_spatial_graph(self):
        """Build the spatial adjacency graph"""
        if self.data is None:
            self.load_data()
            
        self.graph = WaterPixelGraph(self.data)
        self.graph.build_adjacency_graph()
        return self.graph
    
    def calculate_neighbor_features(self):
        """Calculate features based on graph neighbors"""
        print("🔄 Calculating graph-based neighbor features...")
        
        if self.graph is None:
            self.build_spatial_graph()
        
        # Initialize neighbor feature columns
        self.data['neighbor_chl_mean'] = np.nan
        self.data['neighbor_chl_std'] = np.nan
        self.data['neighbor_chl_min'] = np.nan
        self.data['neighbor_chl_max'] = np.nan
        self.data['neighbor_count'] = 0
        
        # Calculate for each observation
        total_rows = len(self.data)
        for idx, row in self.data.iterrows():
            if idx % 1000 == 0:
                print(f"   Processing {idx}/{total_rows}...")
            
            coord_key = f"{row['lon']:.6f}_{row['lat']:.6f}"
            date = row['date']
            
            if coord_key in self.graph.adjacency_graph:
                neighbor_indices = self.graph.adjacency_graph[coord_key]
                
                if len(neighbor_indices) > 0:
                    # Get neighbor coordinates
                    neighbor_coords = self.graph.pixel_coordinates[neighbor_indices]
                    
                    # Find neighbor chlorophyll values for the same date
                    neighbor_values = []
                    for neighbor_coord in neighbor_coords:
                        neighbor_data = self.data[
                            (abs(self.data['lon'] - neighbor_coord[0]) < 1e-5) & 
                            (abs(self.data['lat'] - neighbor_coord[1]) < 1e-5) &
                            (self.data['date'] == date)
                        ]
                        
                        if len(neighbor_data) > 0:
                            chl_value = neighbor_data['chlorophyll_a'].iloc[0]
                            if not pd.isna(chl_value):
                                neighbor_values.append(chl_value)
                    
                    # Calculate neighbor statistics
                    if len(neighbor_values) > 0:
                        self.data.loc[idx, 'neighbor_chl_mean'] = np.mean(neighbor_values)
                        self.data.loc[idx, 'neighbor_chl_std'] = np.std(neighbor_values) if len(neighbor_values) > 1 else 0
                        self.data.loc[idx, 'neighbor_chl_min'] = np.min(neighbor_values)
                        self.data.loc[idx, 'neighbor_chl_max'] = np.max(neighbor_values)
                        self.data.loc[idx, 'neighbor_count'] = len(neighbor_values)
        
        print("✅ Graph-based neighbor features calculated")
        return self.data
    
    def train_graph_enhanced_model(self):
        """Train model with graph-based spatial features"""
        print("🤖 Training graph-enhanced model...")
        
        # Prepare features
        spectral_features = ['Rrs443', 'Rrs490', 'Rrs560', 'Rrs665', 'Rrs705', 'Rrs740', 'Rrs783', 'Rrs865']
        
        # Add temporal features
        self.data['day_of_year'] = self.data['date'].dt.dayofyear
        self.data['month'] = self.data['date'].dt.month
        self.data['season'] = ((self.data['month'] + 2) // 3) % 4 + 1
        temporal_features = ['day_of_year', 'month', 'season']
        
        # Graph-based spatial features
        spatial_features = ['neighbor_chl_mean', 'neighbor_chl_std', 'neighbor_chl_min', 'neighbor_chl_max', 'neighbor_count']
        
        all_features = spectral_features + temporal_features + spatial_features
        
        # Prepare training data
        X = self.data[all_features].values
        y = self.data['chlorophyll_a'].values
        
        # Clean data
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(X).any(axis=1) | np.isinf(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        print(f"   Training on {len(X_clean)} samples")
        print(f"   Features: {len(spectral_features)} spectral + {len(temporal_features)} temporal + {len(spatial_features)} graph-spatial")
        
        # Scale and train
        X_scaled = self.scaler.fit_transform(X_clean)
        
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            random_state=42
        )
        self.model.fit(X_scaled, y_clean)
        
        # Feature importance
        importance = self.model.feature_importances_
        feature_df = pd.DataFrame({
            'feature': all_features,
            'importance': importance,
            'type': (['spectral'] * len(spectral_features) + 
                    ['temporal'] * len(temporal_features) + 
                    ['graph_spatial'] * len(spatial_features))
        }).sort_values('importance', ascending=False)
        
        # Model performance
        score = self.model.score(X_scaled, y_clean)
        
        print("✅ Graph-enhanced model trained!")
        print(f"   Model R²: {score:.3f}")
        
        print("\nFeature importance by type:")
        for feat_type in ['spectral', 'temporal', 'graph_spatial']:
            type_importance = feature_df[feature_df['type'] == feat_type]['importance'].sum()
            print(f"   {feat_type}: {type_importance:.3f}")
        
        print("\nTop spatial features:")
        spatial_features_df = feature_df[feature_df['type'] == 'graph_spatial']
        for _, row in spatial_features_df.iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        return self.model, feature_df
    
    def run_graph_analysis(self):
        """Run complete graph-based analysis"""
        print("🚀 GRAPH-BASED CHLOROPHYLL FORECASTING")
        print("=" * 50)
        
        # Load data and build graph
        self.load_data()
        self.build_spatial_graph()
        
        # Visualize graph
        self.graph.visualize_graph()
        
        # Calculate neighbor features
        self.calculate_neighbor_features()
        
        # Train enhanced model
        model, feature_importance = self.train_graph_enhanced_model()
        
        # Save results
        self.data.to_csv('graph_enhanced_chlorophyll_data.csv', index=False)
        feature_importance.to_csv('graph_feature_importance.csv', index=False)
        
        print(f"\n🎉 GRAPH ANALYSIS COMPLETE!")
        print("Generated files:")
        print("  - water_pixel_graph.png (graph visualization)")
        print("  - graph_enhanced_chlorophyll_data.csv (data with graph features)")
        print("  - graph_feature_importance.csv (feature analysis)")
        
        return self.data, feature_importance

def main():
    """Run graph-based analysis"""
    forecaster = GraphBasedChlorophyllForecaster("chl_connect_timeseries_2000pts.csv")
    data, importance = forecaster.run_graph_analysis()
    return data, importance

if __name__ == "__main__":
    data, importance = main()