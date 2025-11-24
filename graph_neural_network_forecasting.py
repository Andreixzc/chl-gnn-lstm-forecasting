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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import json
import glob
import os
import warnings
warnings.filterwarnings('ignore')

class ChlorophyllGraphDataset:
    """Prepare chlorophyll time series data for Graph Neural Networks"""
    
    def __init__(self, data_dir="daily_snapshots", aoi_path="Area.json"):
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
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_data(self):
        """Load and prepare time series data from daily snapshot files"""
        print("📊 Loading chlorophyll time series data from daily snapshots...")
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
        
        # Sort by date
        self.data = self.data.sort_values('date').reset_index(drop=True)
        
        # Get unique pixel coordinates
        self.pixel_coords = self.data[['lon', 'lat']].drop_duplicates().reset_index(drop=True)
        self.pixel_coords['pixel_id'] = range(len(self.pixel_coords))
        
        print(f"✅ Loaded {len(self.data)} observations from {len(self.pixel_coords)} pixels")
        print(f"📅 Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        print(f"📅 Unique dates: {self.data['date'].dt.date.nunique()}")
        
        return self.data
    
    def build_spatial_adjacency(self, max_distance=0.025):
        """
        Build spatial adjacency matrix for water pixels
        Uses geographic distance to determine neighbors
        """
        print(f"🔗 Building spatial adjacency graph (max_distance={max_distance})...")
        
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
        print(f"✅ Adjacency graph built:")
        print(f"   Total edges: {self.edge_index.shape[1]}")
        print(f"   Avg neighbors per pixel: {neighbor_counts.mean():.1f}")
        print(f"   Min/Max neighbors: {neighbor_counts.min()}/{neighbor_counts.max()}")
        
        return self.edge_index
    
    def create_temporal_sequences(self, sequence_length=8, prediction_steps=6):
        """
        Create temporal sequences for each pixel
        Returns node features matrix: [n_pixels, n_timesteps, n_features]
        """
        print(f"⏱️ Creating temporal sequences (length={sequence_length}, predict={prediction_steps})...")
        
        if self.data is None:
            self.load_data()
        
        # Get unique dates and sort
        dates = sorted(self.data['date'].unique())
        n_dates = len(dates)
        n_pixels = len(self.pixel_coords)
        
        print(f"   Processing {n_dates} dates for {n_pixels} pixels...")
        
        # Create time series matrix: [pixels, dates]
        time_series_matrix = np.full((n_pixels, n_dates), np.nan)
        
        # Fill the matrix
        for _, row in self.data.iterrows():
            # Find pixel ID
            pixel_mask = (
                (self.pixel_coords['lon'] - row['lon']).abs() < 1e-5) & \
                ((self.pixel_coords['lat'] - row['lat']).abs() < 1e-5
            )
            
            if pixel_mask.any():
                pixel_id = self.pixel_coords[pixel_mask].index[0]
                date_id = dates.index(row['date'])
                time_series_matrix[pixel_id, date_id] = row['chlorophyll_a']
        
        # Handle missing values with forward fill then backward fill
        for pixel_id in range(n_pixels):
            pixel_series = pd.Series(time_series_matrix[pixel_id, :])
            pixel_series = pixel_series.fillna(method='ffill').fillna(method='bfill')
            time_series_matrix[pixel_id, :] = pixel_series.values
        
        # Normalize data
        original_shape = time_series_matrix.shape
        flat_data = time_series_matrix.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(flat_data)
        normalized_matrix = scaled_data.reshape(original_shape)
        
        # Create sequences for training
        sequences = []
        targets = []
        
        for t in range(sequence_length, n_dates - prediction_steps + 1):
            # Input sequence: [pixels, sequence_length]
            X_seq = normalized_matrix[:, t-sequence_length:t]
            
            # Target: [pixels, prediction_steps]
            y_seq = normalized_matrix[:, t:t+prediction_steps]
            
            sequences.append(X_seq)
            targets.append(y_seq)
        
        sequences = np.array(sequences)  # [n_sequences, n_pixels, sequence_length]
        targets = np.array(targets)      # [n_sequences, n_pixels, prediction_steps]
        
        print(f"✅ Created {len(sequences)} training sequences")
        print(f"   Input shape: {sequences.shape}")
        print(f"   Target shape: {targets.shape}")
        
        # Store for later use
        self.sequences = sequences
        self.targets = targets
        self.dates = dates
        self.time_series_matrix = time_series_matrix
        
        return sequences, targets
    
    def create_graph_data_objects(self):
        """Create PyTorch Geometric Data objects for training"""
        print("📦 Creating graph data objects...")
        
        if self.edge_index is None:
            self.build_spatial_adjacency()
        
        if not hasattr(self, 'sequences'):
            self.create_temporal_sequences()
        
        graph_data_list = []
        
        for i in range(len(self.sequences)):
            # Node features: [n_pixels, sequence_length]
            x = torch.tensor(self.sequences[i], dtype=torch.float32)
            
            # Targets: [n_pixels, prediction_steps]
            y = torch.tensor(self.targets[i], dtype=torch.float32)
            
            # Create graph data object
            data = Data(
                x=x,
                edge_index=self.edge_index,
                y=y,
                num_nodes=len(self.pixel_coords)
            )
            
            graph_data_list.append(data)
        
        print(f"✅ Created {len(graph_data_list)} graph data objects")
        return graph_data_list

class GraphChlorophyllNet(nn.Module):
    """
    Graph Neural Network for chlorophyll prediction
    Combines spatial graph convolution with temporal LSTM
    """
    
    def __init__(self, input_dim=8, hidden_dim=64, lstm_hidden=32, output_dim=6, num_graph_layers=2):
        super(GraphChlorophyllNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        self.output_dim = output_dim
        
        # Graph convolution layers for spatial modeling
        self.graph_convs = nn.ModuleList()
        self.graph_convs.append(GCNConv(1, hidden_dim))  # Input is single value per time step
        
        for _ in range(num_graph_layers - 1):
            self.graph_convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(lstm_hidden // 2, output_dim)
        )
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_graph_layers)
        ])
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # x shape should be [num_nodes, sequence_length]
        num_nodes = x.size(0)
        sequence_length = x.size(1)
        
        # Process each time step with graph convolution
        temporal_features = []
        
        for t in range(sequence_length):
            # Current time step features: [num_nodes] -> [num_nodes, 1]
            h = x[:, t].unsqueeze(1)  # Add feature dimension
            
            # Apply graph convolutions
            for i, (conv, norm) in enumerate(zip(self.graph_convs, self.batch_norms)):
                h = conv(h, edge_index)  # Keep h as [num_nodes, features]
                h = norm(h)
                h = F.relu(h)
                if i < len(self.graph_convs) - 1:  # Don't apply dropout to last layer
                    h = F.dropout(h, training=self.training)
            
            temporal_features.append(h)
        
        # Stack temporal features: [num_nodes, sequence_length, hidden_dim]
        temporal_features = torch.stack(temporal_features, dim=1)
        
        # Apply LSTM for temporal modeling
        lstm_out, _ = self.lstm(temporal_features)
        
        # Use last LSTM output for prediction
        final_features = lstm_out[:, -1, :]  # [num_nodes, lstm_hidden]
        
        # Generate predictions
        predictions = self.output_layers(final_features)  # [num_nodes, output_dim]
        
        return predictions

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
        print(f"🔧 Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("   ⚠️ GPU not available, using CPU (this will be slower)")
        
    def prepare_data(self, sequence_length=8, prediction_steps=6):
        """Prepare all data for training"""
        print("🔄 Preparing graph data...")
        
        # Load and process data
        self.dataset.load_data()
        self.dataset.build_spatial_adjacency()
        self.dataset.create_temporal_sequences(sequence_length, prediction_steps)
        
        # Create graph data objects
        graph_data_list = self.dataset.create_graph_data_objects()
        
        # Split into train/validation
        split_idx = int(0.8 * len(graph_data_list))
        self.train_data = graph_data_list[:split_idx]
        self.val_data = graph_data_list[split_idx:]
        
        print(f"✅ Data prepared:")
        print(f"   Training samples: {len(self.train_data)}")
        print(f"   Validation samples: {len(self.val_data)}")
        
        return self.train_data, self.val_data
    
    def build_model(self, input_dim=1, hidden_dim=64, lstm_hidden=32, output_dim=6):
        """Build and initialize the graph neural network"""
        print("🏗️ Building Graph Neural Network...")
        
        self.model = GraphChlorophyllNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            lstm_hidden=lstm_hidden,
            output_dim=output_dim
        ).to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Model built:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model device: {next(self.model.parameters()).device}")
        
        return self.model
    
    def train_model(self, epochs=100, lr=0.001, weight_decay=1e-5):
        """Train the graph neural network"""
        print(f"🚀 Training model for {epochs} epochs...")
        
        if self.model is None:
            self.build_model()
        
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
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
            
            # Update learning rate
            scheduler.step(val_loss)
            
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
        
        print(f"✅ Training completed!")
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
        
        print("✅ Training history saved: graph_training_history.png")
    
    def predict_future_maps(self, steps_ahead=6):
        """Generate future chlorophyll prediction maps"""
        print(f"🔮 Generating {steps_ahead} future prediction maps...")
        
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
        original_shape = predictions_np.shape
        flat_predictions = predictions_np.reshape(-1, 1)
        denorm_predictions = self.dataset.scaler.inverse_transform(flat_predictions)
        final_predictions = denorm_predictions.reshape(original_shape)
        
        # Create coordinate mapping for visualization
        coords = self.dataset.pixel_coords[['lon', 'lat']].values
        
        print(f"✅ Generated predictions for {len(coords)} pixels")
        print(f"   Prediction range: {final_predictions.min():.2f} - {final_predictions.max():.2f} mg/m³")
        
        return final_predictions, coords
    
    def visualize_future_maps(self, predictions, coords, save_path="graph_future_maps.png"):
        """Create publication-quality future prediction maps"""
        print("🎨 Creating future prediction maps...")
        
        n_steps = predictions.shape[1]
        
        # Create subplot grid
        cols = 3
        rows = (n_steps + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for step in range(n_steps):
            row, col = step // cols, step % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Get predictions for this time step
            step_predictions = predictions[:, step]
            
            # Create scatter plot
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
            ax.set_title(f'Future Month {step + 1}')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Chlorophyll-a (mg/m³)')
        
        # Hide empty subplots
        for step in range(n_steps, rows * cols):
            if rows > 1:
                row, col = step // cols, step % cols
                axes[row, col].axis('off')
            else:
                if step < len(axes):
                    axes[step].axis('off')
        
        plt.suptitle('Graph Neural Network - Future Chlorophyll Predictions\nTrês Marias Reservoir', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Future maps saved: {save_path}")
    
    def run_complete_analysis(self):
        """Run the complete graph neural network analysis"""
        print("🚀 GRAPH NEURAL NETWORK CHLOROPHYLL FORECASTING")
        print("=" * 60)
        print("READING FROM DAILY SNAPSHOT FILES")
        print("=" * 60)
        
        # Prepare data
        self.prepare_data()
        
        # Build and train model
        self.build_model()
        self.train_model(epochs=50)
        
        # Generate predictions
        predictions, coords = self.predict_future_maps()
        
        # Visualize results
        self.visualize_future_maps(predictions, coords)
        
        # Calculate some statistics
        print(f"\n📊 PREDICTION STATISTICS:")
        print(f"   Pixels predicted: {len(coords)}")
        print(f"   Future time steps: {predictions.shape[1]}")
        print(f"   Chlorophyll range: {predictions.min():.2f} - {predictions.max():.2f} mg/m³")
        print(f"   Mean prediction: {predictions.mean():.2f} mg/m³")
        
        print(f"\n🎉 GRAPH NEURAL NETWORK ANALYSIS COMPLETE!")
        print("Generated files:")
        print("  - best_graph_model.pth (trained model)")
        print("  - graph_training_history.png (training progress)")
        print("  - graph_future_maps.png (prediction maps)")
        
        return predictions, coords

def main():
    """Run graph neural network analysis with daily snapshot files"""
    print("\n" + "="*60)
    print("MODIFIED VERSION - READS FROM DAILY SNAPSHOTS")
    print("="*60 + "\n")
    
    forecaster = GraphChlorophyllForecaster(data_dir="daily_snapshots")
    predictions, coords = forecaster.run_complete_analysis()
    return predictions, coords

if __name__ == "__main__":
    predictions, coords = main()