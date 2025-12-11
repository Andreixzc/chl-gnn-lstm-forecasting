"""
Graph Neural Network for Chlorophyll Spatial-Temporal Forecasting
MODIFIED: Reads from daily snapshot files instead of single CSV
Implements GraphConv + LSTM for reservoir chlorophyll prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import json
import glob
import os
import warnings
import geopandas as gpd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
import contextily as ctx

warnings.filterwarnings('ignore')

class ChlorophyllGraphDataset:
    """Prepare chlorophyll time series data for Graph Neural Networks"""
    
    def __init__(self, data_dir="daily_snapshots_5day", aoi_path="Area.json"):
        """
        Initialize dataset loader
        
        Args:
            data_dir: Directory containing daily snapshot CSV files
            aoi_path: Path to area of interest JSON file
        """
        self.data_dir = data_dir
        self.aoi_path = aoi_path
        self.data = None
        self.pixel_coords = None
        self.adjacency_matrix = None
        self.edge_index = None
        self.scaler = RobustScaler()
        
    def load_data(self):
        """Load and prepare time series data from daily snapshot files"""
        print(" Loading chlorophyll time series data from daily snapshots...")
        print(f"   Reading from directory: {self.data_dir}/")
        
        # Get all CSV files in the directory
        csv_pattern = os.path.join(self.data_dir, "snapshot_*.csv")
        csv_files = sorted(glob.glob(csv_pattern))
        
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No snapshot files found in {self.data_dir}/")
        
        print(f"   Found {len(csv_files)} daily snapshot files")
        
        # Read and concatenate all files
        dataframes = []
        for i, csv_file in enumerate(csv_files):
            if i % 10 == 0:
                print(f"   Loading file {i+1}/{len(csv_files)}...", end='\r')
            df = pd.read_csv(csv_file)
            dataframes.append(df)
        
        print(f"   Loading file {len(csv_files)}/{len(csv_files)}... Done!")
        
        # Concatenate all dataframes
        self.data = pd.concat(dataframes, ignore_index=True)
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # Filter outliers (keep only valid chlorophyll values)
        print(f" Filtering outliers (0 <= chl <= 100)...")
        original_len = len(self.data)
        self.data = self.data[(self.data['chlorophyll_a'] >= 0) & (self.data['chlorophyll_a'] <= 100)]
        print(f"   Removed {original_len - len(self.data)} outliers")

        # Sort by date
        self.data = self.data.sort_values('date').reset_index(drop=True)
        
        # Get unique pixel coordinates
        self.pixel_coords = self.data[['lon', 'lat']].drop_duplicates().reset_index(drop=True)
        self.pixel_coords['pixel_id'] = range(len(self.pixel_coords))
        
        print(f" Loaded {len(self.data)} observations from {len(self.pixel_coords)} pixels")
        print(f" Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        print(f" Unique dates: {self.data['date'].dt.date.nunique()}")
        
        return self.data
    
    def build_spatial_adjacency(self, max_distance=0.025):
        """
        Build spatial adjacency matrix for water pixels
        Uses geographic distance to determine neighbors
        """
        print(f" Building spatial adjacency graph (max_distance={max_distance})...")
        
        n_pixels = len(self.pixel_coords)
        coords = self.pixel_coords[['lon', 'lat']].values
        
        # Calculate pairwise distances
        from scipy.spatial.distance import cdist
        distance_matrix = cdist(coords, coords, metric='euclidean')
        
        # Create adjacency matrix (1 if neighbors, 0 if not)
        self.adjacency_matrix = (distance_matrix <= max_distance) & (distance_matrix > 0)
        
        # Convert to edge index format for PyTorch Geometric
        edge_indices = np.where(self.adjacency_matrix)
        self.edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
        
        # Calculate connectivity statistics
        neighbor_counts = np.sum(self.adjacency_matrix, axis=1)
        print(f" Adjacency graph built:")
        print(f"   Total edges: {self.edge_index.shape[1]}")
        print(f"   Avg neighbors per pixel: {neighbor_counts.mean():.1f}")
        print(f"   Min/Max neighbors: {neighbor_counts.min()}/{neighbor_counts.max()}")
        
        return self.edge_index
    
    def create_temporal_sequences(self, sequence_length=14, prediction_steps=6):
        """Create sequences using Rrs bands as features plus derived spectral indices"""
        print(f" Creating temporal sequences with Rrs features (length={sequence_length})...")
        
        if self.data is None:
            self.load_data()
        
        # Add derived spectral indices for chlorophyll estimation
        print("   Computing spectral indices...")
        # NDCI (Normalized Difference Chlorophyll Index): (Rrs705 - Rrs665) / (Rrs705 + Rrs665)
        if 'Rrs705' in self.data.columns and 'Rrs665' in self.data.columns:
            self.data['NDCI'] = (self.data['Rrs705'] - self.data['Rrs665']) / (self.data['Rrs705'] + self.data['Rrs665'] + 1e-8)
        
        # 2BDA (Two-Band Difference Algorithm): Rrs705 / Rrs665
        if 'Rrs705' in self.data.columns and 'Rrs665' in self.data.columns:
            self.data['2BDA'] = self.data['Rrs705'] / (self.data['Rrs665'] + 1e-8)
        
        # 3BDA (Three-Band Difference Algorithm): (1/Rrs665 - 1/Rrs705) * Rrs740
        if 'Rrs665' in self.data.columns and 'Rrs705' in self.data.columns and 'Rrs740' in self.data.columns:
            self.data['3BDA'] = (1/(self.data['Rrs665']+1e-8) - 1/(self.data['Rrs705']+1e-8)) * self.data['Rrs740']
        
        # Blue-Green ratio (good for clear water)
        if 'Rrs490' in self.data.columns and 'Rrs560' in self.data.columns:
            self.data['BG_ratio'] = self.data['Rrs490'] / (self.data['Rrs560'] + 1e-8)
        
        # Define feature columns
        rrs_bands = ['Rrs443', 'Rrs490', 'Rrs560', 'Rrs665', 'Rrs705', 'Rrs740', 'Rrs783', 'Rrs865']
        
        # Check if Rrs bands exist in data
        available_bands = [col for col in rrs_bands if col in self.data.columns]
        if len(available_bands) < len(rrs_bands):
            print(f" Warning: Some Rrs bands missing. Found: {available_bands}")
            rrs_bands = available_bands
        
        # Add spectral indices to feature list
        spectral_indices = ['NDCI', '2BDA', '3BDA', 'BG_ratio']
        available_indices = [col for col in spectral_indices if col in self.data.columns]
        print(f"   Added spectral indices: {available_indices}")
        all_bands = rrs_bands + available_indices
            
        dates = sorted(self.data['date'].unique())
        n_dates = len(dates)
        n_pixels = len(self.pixel_coords)
        n_features = len(all_bands) + 1  # Rrs bands + spectral indices + chlorophyll
        
        print(f"   Processing {n_dates} dates for {n_pixels} pixels with {n_features} features...")
        
        # Create feature matrix: [pixels, dates, features]
        feature_matrix = np.full((n_pixels, n_dates, n_features), np.nan)
        
        # Create date-to-index mapping for faster lookup
        date_to_idx = {date: idx for idx, date in enumerate(dates)}
        
        # Create pixel coordinate to ID mapping
        pixel_to_id = {}
        for idx, row in self.pixel_coords.iterrows():
            # Use rounded coordinates to match, avoiding float precision issues
            key = (round(row['lon'], 5), round(row['lat'], 5))
            pixel_to_id[key] = idx
        
        # Fill the matrix
        print("   Filling feature matrix...")
        
        # Pivot for chlorophyll
        chl_pivot = self.data.pivot_table(index=['lon', 'lat'], columns='date', values='chlorophyll_a', aggfunc='mean')
        
        # Fill chlorophyll (last feature)
        for (lon, lat), row in chl_pivot.iterrows():
            key = (round(lon, 5), round(lat, 5))
            if key in pixel_to_id:
                pid = pixel_to_id[key]
                valid_dates = row.dropna().index
                indices = [date_to_idx[d] for d in valid_dates]
                feature_matrix[pid, indices, -1] = row[valid_dates].values

        # Fill Rrs bands and spectral indices
        for i, band in enumerate(all_bands):
            band_pivot = self.data.pivot_table(index=['lon', 'lat'], columns='date', values=band, aggfunc='mean')
            for (lon, lat), row in band_pivot.iterrows():
                key = (round(lon, 5), round(lat, 5))
                if key in pixel_to_id:
                    pid = pixel_to_id[key]
                    valid_dates = row.dropna().index
                    indices = [date_to_idx[d] for d in valid_dates]
                    feature_matrix[pid, indices, i] = row[valid_dates].values
        
        # Handle missing values - interpolate per pixel per feature
        print("   Interpolating missing values...")
        for pixel_id in range(n_pixels):
            for feat_id in range(n_features):
                series = pd.Series(feature_matrix[pixel_id, :, feat_id])
                series = series.interpolate(method='linear', limit=5)
                series = series.fillna(method='ffill', limit=3).fillna(method='bfill', limit=3)
                feature_matrix[pixel_id, :, feat_id] = series.values
        
        # Normalize each feature separately
        print("   Normalizing features...")
        self.scalers = {}
        normalized_matrix = np.zeros_like(feature_matrix)
        
        feature_names = all_bands + ['chlorophyll_a']
        for feat_id, feat_name in enumerate(feature_names):
            # Flatten, fit, transform, reshape
            feat_data = feature_matrix[:, :, feat_id].reshape(-1, 1)
            
            # Handle any remaining NaNs before scaling (replace with mean)
            if np.isnan(feat_data).any():
                nan_mask = np.isnan(feat_data)
                mean_val = np.nanmean(feat_data)
                feat_data[nan_mask] = mean_val
                
            scaler = RobustScaler()
            normalized_matrix[:, :, feat_id] = scaler.fit_transform(feat_data).reshape(n_pixels, n_dates)
            self.scalers[feat_name] = scaler
        
        # Keep chlorophyll scaler as main scaler for inverse transform compatibility
        self.scaler = self.scalers['chlorophyll_a']
        
        # Create sequences
        sequences = []
        targets = []
        
        print("   Generating sequences...")
        for t in range(sequence_length, n_dates - prediction_steps + 1):
            # Input: [n_pixels, sequence_length, n_features]
            X_seq = normalized_matrix[:, t-sequence_length:t, :]
            
            # Target: [n_pixels, prediction_steps] - only chlorophyll (last feature)
            y_seq = normalized_matrix[:, t:t+prediction_steps, -1]
            
            # Skip sequences with too many NaN values (if any remain)
            if np.isnan(X_seq).mean() < 0.3 and np.isnan(y_seq).mean() < 0.1:
                # Fill remaining NaNs with 0 (after normalization, 0  median)
                X_seq = np.nan_to_num(X_seq, nan=0.0)
                y_seq = np.nan_to_num(y_seq, nan=0.0)
                
                sequences.append(X_seq)
                targets.append(y_seq)
        
        sequences = np.array(sequences)  # [n_sequences, n_pixels, seq_len, n_features]
        targets = np.array(targets)      # [n_sequences, n_pixels, prediction_steps]
        
        print(f" Created {len(sequences)} training sequences")
        print(f"   Input shape: {sequences.shape}")
        print(f"   Target shape: {targets.shape}")
        print(f"   Features: {feature_names}")
        
        # Store for later use
        self.sequences = sequences
        self.targets = targets
        self.dates = dates
        self.n_features = n_features
        self.feature_names = feature_names
        
        return sequences, targets
    
    def create_graph_data_objects(self):
        """Create PyTorch Geometric Data objects for training"""
        print(" Creating graph data objects...")
        
        if self.edge_index is None:
            self.build_spatial_adjacency()
        
        if not hasattr(self, 'sequences'):
            self.create_temporal_sequences()
        
        graph_data_list = []
        
        for i in range(len(self.sequences)):
            # Node features: [n_pixels, sequence_length, n_features]
            x = torch.tensor(self.sequences[i], dtype=torch.float32)
            
            # Targets: [n_pixels, prediction_steps]
            y = torch.tensor(self.targets[i], dtype=torch.float32)
            
            data = Data(
                x=x,
                edge_index=self.edge_index,
                y=y,
                num_nodes=len(self.pixel_coords)
            )
            
            graph_data_list.append(data)
        
        print(f" Created {len(graph_data_list)} graph data objects")
        return graph_data_list

class TemporalAttention(nn.Module):
    """Attention mechanism to weight temporal steps"""
    
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )
    
    def forward(self, lstm_output):
        # lstm_output: [num_nodes, seq_len, hidden_dim]
        attn_weights = self.attention(lstm_output)  # [num_nodes, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        # Weighted sum
        context = torch.sum(lstm_output * attn_weights, dim=1)  # [num_nodes, hidden_dim]
        return context, attn_weights


class GraphChlorophyllNet(nn.Module):
    """
    Improved GCN + LSTM for chlorophyll prediction
    Uses multiple Rrs bands as input features with Residual Connections, Bi-LSTM and Attention
    """
    
    def __init__(self, input_dim=9, hidden_dim=128, lstm_hidden=128, output_dim=6, num_graph_layers=3, dropout=0.2):
        super(GraphChlorophyllNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        self.output_dim = output_dim
        
        # Input projection: map input features to hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # GELU often works better than ReLU
            nn.Dropout(dropout)
        )
        
        # Graph convolution layers with residual connections
        self.graph_convs = nn.ModuleList()
        self.graph_norms = nn.ModuleList()
        
        for _ in range(num_graph_layers):
            self.graph_convs.append(GCNConv(hidden_dim, hidden_dim))
            self.graph_norms.append(nn.LayerNorm(hidden_dim))
            
        self.gcn_dropout = nn.Dropout(dropout)
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Temporal attention mechanism
        self.temporal_attention = TemporalAttention(lstm_hidden * 2)
        
        # Output layers with skip connection from attention
        # Input to output layers is lstm_hidden * 2 (because bidirectional)
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_hidden * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # x shape: [num_nodes, sequence_length, input_dim]
        num_nodes = x.size(0)
        sequence_length = x.size(1)
        
        temporal_features = []
        
        # Process each time step
        for t in range(sequence_length):
            # Get features for this time step: [num_nodes, input_dim]
            h = x[:, t, :]
            
            # Project to hidden dimension: [num_nodes, hidden_dim]
            h = self.input_proj(h)
            
            # Apply graph convolutions with residual connections
            for conv, norm in zip(self.graph_convs, self.graph_norms):
                h_res = h
                h = conv(h, edge_index)
                h = norm(h + h_res)  # Residual connection
                h = F.relu(h)
                h = self.gcn_dropout(h)
            
            temporal_features.append(h)
        
        # Stack temporal features: [num_nodes, sequence_length, hidden_dim]
        temporal_features = torch.stack(temporal_features, dim=1)
        
        # Apply LSTM for temporal modeling
        # lstm_out: [num_nodes, sequence_length, lstm_hidden * 2]
        lstm_out, _ = self.lstm(temporal_features)
        
        # Apply temporal attention instead of just using last output
        context, attn_weights = self.temporal_attention(lstm_out)
        
        # Combine attention context with last LSTM output for richer representation
        last_output = lstm_out[:, -1, :]
        final_features = context + last_output  # Residual connection
        
        # Generate predictions
        predictions = self.output_layers(final_features)  # [num_nodes, output_dim]
        
        return predictions

class SatelliteMapGenerator:
    """
    Generates satellite overlay maps using the same approach as CompleteMapGenerator
    """
    
    def __init__(self, aoi_path="Area.json"):
        self.aoi_path = aoi_path
        self.gdf = None
        self.bounds = None
        self.crs = "EPSG:4326"
        
        # Load boundaries
        self.load_boundaries()
    
    def load_boundaries(self):
        """Load vector boundaries"""
        if os.path.exists(self.aoi_path):
            self.gdf = gpd.read_file(self.aoi_path)
            self.bounds = self.gdf.total_bounds  # [minx, miny, maxx, maxy]
            print(f" Loaded Boundaries: {self.aoi_path}")
        else:
            raise FileNotFoundError(f" Could not find {self.aoi_path}")
    
    def generate_smooth_surface(self, points, values, width=1500):
        """
        Generate high-res smooth surface from point data
        
        Args:
            points: numpy array of shape (n, 2) with [lon, lat] coordinates
            values: numpy array of shape (n,) with chlorophyll values
            width: width of output image in pixels
            
        Returns:
            grid_img: 2D numpy array with smooth chlorophyll surface
            transform: rasterio transform for georeferencing
            extent: [minx, maxx, miny, maxy] for matplotlib plotting
        """
        minx, miny, maxx, maxy = self.bounds
        pixel_size = (maxx - minx) / width
        
        # Calculate height
        height = int(np.ceil((maxy - miny) / pixel_size))
        actual_miny = maxy - (height * pixel_size)
        
        # Save dimensions
        shape = (height, width)
        extent = [minx, maxx, actual_miny, maxy]
        
        # Generate grid centers
        grid_x = np.linspace(minx + pixel_size/2, maxx - pixel_size/2, width)
        grid_y = np.linspace(actual_miny + pixel_size/2, maxy - pixel_size/2, height)
        grid_x_mesh, grid_y_mesh = np.meshgrid(grid_x, grid_y)
        
        # 1. Cubic interpolation (structure)
        grid_cubic = griddata(points, values, (grid_x_mesh, grid_y_mesh), method='cubic')
        
        # 2. Nearest neighbor (filler)
        grid_near = griddata(points, values, (grid_x_mesh, grid_y_mesh), method='nearest')
        
        # 3. Combine
        grid_final = np.where(np.isnan(grid_cubic), grid_near, grid_cubic)
        
        # 4. Flip & blur (organic fluid look)
        grid_img = np.flipud(grid_final)
        grid_img = gaussian_filter(grid_img, sigma=2)
        
        # 5. Apply water mask
        transform = from_origin(minx, maxy, pixel_size, pixel_size)
        
        mask = rasterize(
            [(geom, 1) for geom in self.gdf.geometry],
            out_shape=shape,
            transform=transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )
        
        # Set land to NaN
        grid_img[mask == 0] = np.nan
        
        return grid_img, transform, extent, shape
    
    def save_satellite_overlay_map(self, points, values, output_path, title_suffix=""):
        """
        Generate and save a satellite overlay map
        
        Args:
            points: numpy array of shape (n, 2) with [lon, lat] coordinates
            values: numpy array of shape (n,) with chlorophyll values
            output_path: path to save the PNG file
            title_suffix: optional suffix to add to the title (e.g., "Step 1")
        """
        print(f" Generating satellite overlay map: {output_path}")
        
        # Generate smooth surface
        grid_img, transform, extent, shape = self.generate_smooth_surface(points, values, width=1500)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # 1. Plot the smooth raster (alpha=0.7 for semi-transparency)
        im = ax.imshow(
            grid_img,
            extent=extent,
            origin='upper',
            cmap='RdYlGn_r',
            alpha=0.7,
            zorder=2  # On top of map
        )
        
        # 2. Add vector outline
        self.gdf.plot(
            ax=ax,
            facecolor='none',
            edgecolor='black',
            linewidth=0.5,
            alpha=0.5,
            zorder=3
        )
        
        # 3. Add satellite basemap using Contextily
        try:
            ctx.add_basemap(
                ax,
                crs=self.crs,
                source=ctx.providers.Esri.WorldImagery,
                zoom=13
            )
            print("    Added Esri World Imagery")
        except Exception as e:
            print(f"    Could not fetch tiles: {e}")
        
        # Formatting
        if title_suffix:
            title = f"Predicted Chlorophyll-a Intensity - {title_suffix}"
        else:
            title = "Predicted Chlorophyll-a Intensity"
        
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.axis('off')  # Clean look
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('mg/m')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f" Saved: {output_path}")

class GraphChlorophyllForecaster:
    """Main forecasting class that orchestrates the entire pipeline"""
    
    def __init__(self, data_dir="daily_snapshots", aoi_path="Area.json"):
        """
        Initialize forecaster
        
        Args:
            data_dir: Directory containing daily snapshot CSV files
            aoi_path: Path to area of interest JSON file
        """
        self.data_dir = data_dir
        self.aoi_path = aoi_path
        self.dataset = ChlorophyllGraphDataset(data_dir, aoi_path)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f" Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("    GPU not available, using CPU (this will be slower)")
        
    def prepare_data(self, sequence_length=5, prediction_steps=1):
        """Prepare all data for training"""
        print(" Preparing graph data...")
        
        # Load and process data
        self.dataset.load_data()
        # Increased max_distance for denser graph connectivity
        self.dataset.build_spatial_adjacency(max_distance=0.035)
        self.dataset.create_temporal_sequences(sequence_length, prediction_steps)
        
        # Create graph data objects
        graph_data_list = self.dataset.create_graph_data_objects()
        
        # Split into train/validation
        split_idx = int(0.8 * len(graph_data_list))
        self.train_data = graph_data_list[:split_idx]
        self.val_data = graph_data_list[split_idx:]
        
        print(f" Data prepared:")
        print(f"   Training samples: {len(self.train_data)}")
        print(f"   Validation samples: {len(self.val_data)}")
        
        return self.train_data, self.val_data
    
    def build_model(self, input_dim=9, hidden_dim=128, lstm_hidden=128, output_dim=6):
        """Build and initialize the graph neural network"""
        print(" Building Graph Neural Network...")
        
        # Check if dataset has features defined to set input_dim correctly
        if hasattr(self.dataset, 'n_features'):
            input_dim = self.dataset.n_features
            print(f"   Using {input_dim} input features from dataset")
        
        self.model = GraphChlorophyllNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            lstm_hidden=lstm_hidden,
            output_dim=output_dim
        ).to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f" Model built:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model device: {next(self.model.parameters()).device}")
        
        return self.model
    
    def train_model(self, epochs=100, lr=0.001, weight_decay=1e-5):
        """Train the graph neural network with improved training strategy"""
        print(f" Training model for {epochs} epochs...")
        
        if self.model is None:
            self.build_model()
        
        # Setup training with AdamW optimizer (better weight decay handling)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Cosine annealing with warm restarts for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        # Huber loss - more robust to outliers than MSE
        criterion = nn.HuberLoss(delta=1.0)
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 30  # Early stopping patience
        patience_counter = 0
        
        # Warmup epochs
        warmup_epochs = 10
        warmup_factor = 0.1
        
        for epoch in range(epochs):
            # Warmup learning rate
            if epoch < warmup_epochs:
                warmup_lr = lr * (warmup_factor + (1 - warmup_factor) * epoch / warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for data in self.train_data:
                data = data.to(self.device)
                optimizer.zero_grad()
                
                predictions = self.model(data)
                loss = criterion(predictions, data.y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(self.train_data)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for data in self.val_data:
                    data = data.to(self.device)
                    predictions = self.model(data)
                    loss = criterion(predictions, data.y)
                    val_loss += loss.item()
            
            val_loss /= len(self.val_data)
            
            # Update learning rate (after warmup)
            if epoch >= warmup_epochs:
                scheduler.step()
            
            # Store losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_graph_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_graph_model.pth'))
        
        print(f" Training completed!")
        print(f"   Best validation loss: {best_val_loss:.6f}")
        
        # Plot training history
        self.plot_training_history(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def plot_training_history(self, train_losses, val_losses):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', alpha=0.8)
        plt.plot(val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Graph Neural Network Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.savefig('graph_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(" Training history saved: graph_training_history.png")
    
    def predict_future_maps(self, steps_ahead=2):
        """Generate future chlorophyll prediction maps"""
        print(f" Generating {steps_ahead} future prediction maps...")
        
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        
        # Use the last sequence from training data
        last_data = self.val_data[-1].to(self.device)
        
        with torch.no_grad():
            predictions = self.model(last_data)  # [num_nodes, prediction_steps]
        
        # Convert back to original scale
        predictions_np = predictions.cpu().numpy()
        
        # Denormalize
        # Handle scaler expecting multiple features
        n_features = self.dataset.scaler.n_features_in_
        
        def denorm_target(arr):
            flat = arr.flatten()
            dummy = np.zeros((flat.size, n_features))
            dummy[:, -1] = flat
            return self.dataset.scaler.inverse_transform(dummy)[:, -1].reshape(arr.shape)

        final_predictions = denorm_target(predictions_np)
        
        # Create coordinate mapping for visualization
        coords = self.dataset.pixel_coords[['lon', 'lat']].values
        
        print(f" Generated predictions for {len(coords)} pixels")
        print(f"   Prediction range: {final_predictions.min():.2f} - {final_predictions.max():.2f} mg/m")
        
        return final_predictions, coords
    
    def export_predictions_to_csv(self, predictions, coords, output_dir='predictions_csv'):
        """
        Export predictions to CSV files (one file per time step)
        
        Args:
            predictions: Prediction array, shape (n_pixels, n_timesteps)
            coords: Coordinate array, shape (n_pixels, 2) [lon, lat]
            output_dir: Directory to save CSV files
        """
        print(f"\n Exporting predictions to CSV files...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        n_steps = predictions.shape[1]
        
        for step in range(n_steps):
            # Create dataframe for this time step
            df = pd.DataFrame({
                'lon': coords[:, 0],
                'lat': coords[:, 1],
                'chlorophyll_a_predicted': predictions[:, step],
                'time_step': step + 1
            })
            
            # Sort by coordinates for easier reading
            df = df.sort_values(['lat', 'lon']).reset_index(drop=True)
            
            # Save to CSV
            filename = os.path.join(output_dir, f'prediction_step_{step+1}.csv')
            df.to_csv(filename, index=False)
            
            print(f"    Saved: prediction_step_{step+1}.csv ({len(df)} pixels)")
        
        # Also create a combined file with all time steps
        all_predictions = []
        for step in range(n_steps):
            df = pd.DataFrame({
                'lon': coords[:, 0],
                'lat': coords[:, 1],
                'chlorophyll_a_predicted': predictions[:, step],
                'time_step': step + 1
            })
            all_predictions.append(df)
        
        combined_df = pd.concat(all_predictions, ignore_index=True)
        combined_filename = os.path.join(output_dir, 'all_predictions_combined.csv')
        combined_df.to_csv(combined_filename, index=False)
        
        print(f"    Saved: all_predictions_combined.csv (all {n_steps} time steps)")
        print(f"\n All predictions exported to '{output_dir}/'")
        
        # Print summary statistics
        print(f"\n Prediction Summary:")
        print(f"   Total pixels: {len(coords)}")
        print(f"   Time steps: {n_steps}")
        print(f"   Total predictions: {len(coords) * n_steps}")
        print(f"   Chlorophyll range: {predictions.min():.2f} - {predictions.max():.2f} mg/m")
        print(f"   Mean: {predictions.mean():.2f} mg/m")
        print(f"   Std Dev: {predictions.std():.2f} mg/m")
        
        return output_dir
    
    def visualize_future_maps(self, predictions, coords, save_path="graph_future_maps.png"):
        """Create publication-quality future prediction maps using SatelliteMapGenerator"""
        print(" Creating future prediction maps...")
        
        n_steps = predictions.shape[1]
        
        # Create output directory for individual maps
        os.makedirs("satellite_maps", exist_ok=True)
        
        # Initialize map generator
        map_gen = None
        try:
            map_gen = SatelliteMapGenerator(aoi_path=self.aoi_path)
        except Exception as e:
            print(f" Failed to initialize map generator: {e}")
        
        # Create subplot grid for the combined figure (using simple scatter for summary)
        cols = 2
        rows = (n_steps + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 6*rows))
        
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for step in range(n_steps):
            # Get predictions for this time step
            step_predictions = predictions[:, step]
            
            # 1. Plot on the combined figure (Scatter)
            ax = axes[step]
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=step_predictions,
                cmap='RdYlGn_r',
                s=60, alpha=0.8,
                vmin=predictions.min(),
                vmax=predictions.max()
            )
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'Future Step {step + 1}')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Chlorophyll-a (mg/m)')
            
            # 2. Generate High-Res Satellite Overlay Map
            if map_gen:
                indiv_save_path = os.path.join("satellite_maps", f"satellite_overlay_step_{step+1}.png")
                try:
                    map_gen.save_satellite_overlay_map(
                        points=coords,
                        values=step_predictions,
                        output_path=indiv_save_path,
                        title_suffix=f"Step {step+1}"
                    )
                    print(f"    Saved satellite overlay: {indiv_save_path}")
                except Exception as e:
                    print(f"    Failed to save overlay for step {step+1}: {e}")
            else:
                # Fallback to simple scatter if map generator fails
                indiv_save_path = os.path.join("satellite_maps", f"prediction_map_step_{step+1}.png")
                plt.figure(figsize=(10, 8))
                plt.scatter(
                    coords[:, 0], coords[:, 1],
                    c=step_predictions,
                    cmap='RdYlGn_r',
                    s=80, alpha=0.8,
                    vmin=predictions.min(),
                    vmax=predictions.max()
                )
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.title(f'Future Prediction - Step {step + 1}\nTrs Marias Reservoir')
                plt.colorbar(label='Chlorophyll-a (mg/m)')
                plt.grid(True, alpha=0.3)
                plt.savefig(indiv_save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    Saved simple map: {indiv_save_path}")
        
        # Hide empty subplots in combined figure
        for i in range(n_steps, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Graph Neural Network - Future Chlorophyll Predictions\nTrs Marias Reservoir', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Future maps saved: {save_path}")
    
    def run_complete_analysis(self):
        """Run the complete graph neural network analysis"""
        print(" GRAPH NEURAL NETWORK CHLOROPHYLL FORECASTING")
        print("=" * 60)
        print("READING FROM DAILY SNAPSHOT FILES")
        print("=" * 60)
        
        # Prepare data
        self.prepare_data()
        
        # Determine output dimension from prepared data
        output_dim = self.dataset.targets[0].shape[1]
        print(f"   Detected output dimension: {output_dim}")
        
        # Build and train model with improved architecture
        self.build_model(hidden_dim=256, lstm_hidden=128, output_dim=output_dim)
        self.train_model(epochs=200, lr=0.002, weight_decay=1e-4)
        
        # Run model evaluation
        print("\n Running Model Evaluation...")
        try:
            from model_evaluation import ModelEvaluator
            from model_visualizations import ModelVisualizations
            
            evaluator = ModelEvaluator()
            visualizer = ModelVisualizations(output_dir='evaluation_figures')
            
            # Collect all validation predictions
            self.model.eval()
            all_preds = []
            all_targets = []
            
            print(f"   Evaluating on {len(self.val_data)} validation sequences...")
            
            with torch.no_grad():
                for data in self.val_data:
                    data = data.to(self.device)
                    pred = self.model(data)
                    target = data.y
                    
                    all_preds.append(pred.cpu().numpy())
                    all_targets.append(target.cpu().numpy())
            
            # Concatenate
            y_pred = np.concatenate(all_preds, axis=0)
            y_true = np.concatenate(all_targets, axis=0)
            
            # Denormalize
            # Handle scaler expecting multiple features
            n_features = self.dataset.scaler.n_features_in_
            
            def denorm_target(arr):
                flat = arr.flatten()
                dummy = np.zeros((flat.size, n_features))
                dummy[:, -1] = flat
                return self.dataset.scaler.inverse_transform(dummy)[:, -1].reshape(arr.shape)

            pred_denorm = denorm_target(y_pred)
            true_denorm = denorm_target(y_true)
            
            # Calculate metrics
            evaluator.calculate_all_metrics(true_denorm, pred_denorm)
            
            # Print metrics summary
            evaluator.print_metrics_summary()
            
            # Save metrics to JSON
            evaluator.save_to_json('evaluation_results/metrics.json')
            
            # Generate evaluation figures
            print("\n Generating evaluation figures...")
            
            # Get metrics for visualization
            metrics = evaluator.metrics['overall']
            
            # Generate predicted vs actual plot
            visualizer.plot_predicted_vs_actual(
                true_denorm, pred_denorm, metrics, 
                save_name='predicted_vs_actual.png'
            )
            
            # Generate residual analysis plot
            visualizer.plot_residuals(
                true_denorm, pred_denorm, 
                save_name='residual_analysis.png'
            )
            
            # Generate per-timestep metrics if available
            if hasattr(evaluator, 'per_timestep_metrics') and evaluator.per_timestep_metrics:
                visualizer.plot_forecast_horizon_performance(
                    evaluator.per_timestep_metrics,
                    save_name='forecast_horizon_performance.png'
                )
                
                # Generate comprehensive summary figure
                visualizer.create_summary_figure(
                    true_denorm, pred_denorm, metrics, 
                    evaluator.per_timestep_metrics,
                    save_name='summary_figure.png'
                )
            
            print(" All evaluation figures generated in 'evaluation_figures/' directory")
            
        except Exception as e:
            print(f" Model evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Generate predictions
        predictions, coords = self.predict_future_maps()
        
        # Export predictions to CSV
        self.export_predictions_to_csv(predictions, coords)
        
        # Visualize discrete results
        self.visualize_future_maps(predictions, coords)
        
        # Calculate some statistics
        print(f"\n PREDICTION STATISTICS:")
        print(f"   Pixels predicted: {len(coords)}")
        print(f"   Future time steps: {predictions.shape[1]}")
        print(f"   Chlorophyll range: {predictions.min():.2f} - {predictions.max():.2f} mg/m")
        print(f"   Mean prediction: {predictions.mean():.2f} mg/m")
        
        print(f"\n GRAPH NEURAL NETWORK ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Generated files:")
        print("=" * 60)
        print("Model:")
        print("   best_graph_model.pth (trained model)")
        print("\nEvaluation results:")
        print("   evaluation_results/metrics.json (all metrics)")
        print("\nTraining visualization:")
        print("   graph_training_history.png (training progress)")
        print("\nEvaluation figures:")
        print("   evaluation_figures/predicted_vs_actual.png")
        print("   evaluation_figures/residual_analysis.png")
        print("   evaluation_figures/forecast_horizon_performance.png")
        print("   evaluation_figures/summary_figure.png")
        print("\nPrediction data (CSV):")
        print("   predictions_csv/prediction_step_1.csv")
        print("   predictions_csv/prediction_step_2.csv")
        print("   ... (one file per time step)")
        print("   predictions_csv/all_predictions_combined.csv")
        print("\nPrediction visualizations (PNG):")
        print("   graph_future_maps.png (discrete point predictions)")
        print("   satellite_maps/satellite_overlay_step_1.png")
        print("   satellite_maps/satellite_overlay_step_2.png")
        print("   ... (one satellite overlay per time step)")
        print("=" * 60)
        
        return predictions, coords

def main():
    """Run graph neural network analysis with daily snapshot files"""
    print("\n" + "="*60)
    print("MODIFIED VERSION - READS FROM DAILY SNAPSHOTS")
    print("="*60 + "\n")
    
    # Use the interpolated dataset
    forecaster = GraphChlorophyllForecaster(data_dir="daily_snapshots_5day")
    predictions, coords = forecaster.run_complete_analysis()
    return predictions, coords

if __name__ == "__main__":
    predictions, coords = main()    