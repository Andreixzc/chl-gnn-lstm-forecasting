"""
Model Visualizations - Performance Plots for Chlorophyll Prediction
Creates publication-quality figures for model evaluation
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class ModelVisualizations:
    """Generate professional visualizations for model evaluation"""
    
    def __init__(self, output_dir='evaluation_figures'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_predicted_vs_actual(self, y_true, y_pred, metrics, save_name='predicted_vs_actual.png'):
        """
        Create predicted vs actual scatter plot with regression line
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            metrics: Dictionary with R2, RMSE, etc.
            save_name: Output filename
        """
        # Flatten arrays and remove NaN
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        y_true_clean = y_true_flat[mask]
        y_pred_clean = y_pred_flat[mask]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot with transparency
        ax.scatter(y_true_clean, y_pred_clean, alpha=0.3, s=20, 
                  label='Predictions', color='steelblue')
        
        # Perfect prediction line (diagonal)
        min_val = min(y_true_clean.min(), y_pred_clean.min())
        max_val = max(y_true_clean.max(), y_pred_clean.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'k--', lw=2, label='Perfect Prediction', alpha=0.7)
        
        # Add regression line
        z = np.polyfit(y_true_clean, y_pred_clean, 1)
        p = np.poly1d(z)
        ax.plot(y_true_clean, p(y_true_clean), 
               'r-', lw=2, alpha=0.7, label='Linear Fit')
        
        # Labels and title
        ax.set_xlabel('Actual Chlorophyll-a (mg/m3)', fontweight='bold')
        ax.set_ylabel('Predicted Chlorophyll-a (mg/m3)', fontweight='bold')
        ax.set_title('Model Performance: Predicted vs Actual', 
                    fontweight='bold', fontsize=14)
        
        # Add metrics text box
        textstr = f"R2 = {metrics['r2']:.4f}\n"
        textstr += f"RMSE = {metrics['rmse']:.4f} mg/m3\n"
        textstr += f"MAE = {metrics['mae']:.4f} mg/m3\n"
        textstr += f"MAPE = {metrics['mape']:.2f}%\n"
        textstr += f"N = {metrics['n_samples']:,}"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', bbox=props)
        
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        output_path = f"{self.output_dir}/{save_name}"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")
        return output_path
    
    def plot_residuals(self, y_true, y_pred, save_name='residual_analysis.png'):
        """Create residual plots for error analysis"""
        # Flatten and clean
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        y_true_clean = y_true_flat[mask]
        y_pred_clean = y_pred_flat[mask]
        
        residuals = y_pred_clean - y_true_clean
        
        # Create 2-panel figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel 1: Residuals vs Predicted
        ax1.scatter(y_pred_clean, residuals, alpha=0.3, s=20, color='steelblue')
        ax1.axhline(y=0, color='r', linestyle='--', lw=2, alpha=0.7)
        ax1.set_xlabel('Predicted Chlorophyll-a (mg/m3)', fontweight='bold')
        ax1.set_ylabel('Residuals (Predicted - Actual)', fontweight='bold')
        ax1.set_title('Residual Plot', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add mean residual line
        mean_residual = residuals.mean()
        ax1.axhline(y=mean_residual, color='orange', linestyle='-', 
                   lw=2, alpha=0.7, label=f'Mean: {mean_residual:.4f}')
        ax1.legend()
        
        # Panel 2: Residual distribution
        ax2.hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', lw=2, alpha=0.7)
        ax2.set_xlabel('Residuals (mg/m3)', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Residual Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        textstr = f"Mean: {residuals.mean():.4f}\n"
        textstr += f"Std: {residuals.std():.4f}\n"
        textstr += f"Skew: {pd.Series(residuals).skew():.4f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax2.text(0.70, 0.95, textstr, transform=ax2.transAxes, 
                fontsize=9, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        output_path = f"{self.output_dir}/{save_name}"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")
        return output_path
    
    def plot_forecast_horizon_performance(self, per_timestep_metrics, 
                                         save_name='forecast_horizon_performance.png'):
        """Plot how model performance degrades over forecast horizon"""
        timesteps = list(per_timestep_metrics.keys())
        r2_values = [per_timestep_metrics[t]['r2'] for t in timesteps]
        rmse_values = [per_timestep_metrics[t]['rmse'] for t in timesteps]
        mae_values = [per_timestep_metrics[t]['mae'] for t in timesteps]
        
        # Create 3-panel figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        x_pos = range(len(timesteps))
        
        # R2 over time
        ax1.plot(x_pos, r2_values, 'o-', linewidth=2, markersize=8, color='steelblue')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(timesteps, rotation=45, ha='right')
        ax1.set_ylabel('R2 Score', fontweight='bold')
        ax1.set_xlabel('Forecast Horizon', fontweight='bold')
        ax1.set_title('R2 by Forecast Step', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # RMSE over time
        ax2.plot(x_pos, rmse_values, 'o-', linewidth=2, markersize=8, color='coral')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(timesteps, rotation=45, ha='right')
        ax2.set_ylabel('RMSE (mg/m3)', fontweight='bold')
        ax2.set_xlabel('Forecast Horizon', fontweight='bold')
        ax2.set_title('RMSE by Forecast Step', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # MAE over time
        ax3.plot(x_pos, mae_values, 'o-', linewidth=2, markersize=8, color='forestgreen')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(timesteps, rotation=45, ha='right')
        ax3.set_ylabel('MAE (mg/m3)', fontweight='bold')
        ax3.set_xlabel('Forecast Horizon', fontweight='bold')
        ax3.set_title('MAE by Forecast Step', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f"{self.output_dir}/{save_name}"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")
        return output_path
    
    def plot_time_series_comparison(self, dates, actual_values, predicted_values, 
                                    pixel_idx=0, save_name='time_series_comparison.png'):
        """
        Plot time series comparison for a specific pixel
        
        Args:
            dates: List of dates
            actual_values: Actual time series [timesteps]
            predicted_values: Predicted time series [timesteps]
            pixel_idx: Which pixel to plot
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.plot(dates, actual_values, 'o-', label='Actual (CHL-CONNECT)', 
               linewidth=2, markersize=6, color='steelblue')
        ax.plot(dates, predicted_values, 's--', label='Predicted (GNN)', 
               linewidth=2, markersize=6, color='coral', alpha=0.7)
        
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Chlorophyll-a (mg/m3)', fontweight='bold')
        ax.set_title(f'Time Series Comparison - Pixel {pixel_idx}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        output_path = f"{self.output_dir}/{save_name}"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")
        return output_path
    
    def plot_training_history(self, train_losses, val_losses, 
                             save_name='training_history.png'):
        """Plot training and validation loss curves"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, label='Training Loss', 
               linewidth=2, color='steelblue')
        ax.plot(epochs, val_losses, label='Validation Loss', 
               linewidth=2, color='coral')
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss (MSE)', fontweight='bold')
        ax.set_title('Model Training History', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Mark best validation loss
        best_epoch = np.argmin(val_losses) + 1
        best_val_loss = min(val_losses)
        ax.plot(best_epoch, best_val_loss, 'r*', markersize=15, 
               label=f'Best (Epoch {best_epoch})')
        ax.legend()
        
        plt.tight_layout()
        output_path = f"{self.output_dir}/{save_name}"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")
        return output_path
    
    def create_summary_figure(self, y_true, y_pred, metrics, per_timestep_metrics,
                             save_name='summary_figure.png'):
        """
        Create comprehensive multi-panel summary figure
        """
        # Flatten and clean data
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        y_true_clean = y_true_flat[mask]
        y_pred_clean = y_pred_flat[mask]
        residuals = y_pred_clean - y_true_clean
        
        # Create figure with gridspec
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Predicted vs Actual (large, top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(y_true_clean, y_pred_clean, alpha=0.3, s=15, color='steelblue')
        min_val = min(y_true_clean.min(), y_pred_clean.min())
        max_val = max(y_true_clean.max(), y_pred_clean.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.7)
        ax1.set_xlabel('Actual (mg/m3)', fontweight='bold')
        ax1.set_ylabel('Predicted (mg/m3)', fontweight='bold')
        ax1.set_title('(A) Predicted vs Actual', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # Add R2 annotation
        ax1.text(0.05, 0.95, f"R2 = {metrics['r2']:.4f}", 
                transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Residual plot (top-middle)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(y_pred_clean, residuals, alpha=0.3, s=15, color='coral')
        ax2.axhline(y=0, color='r', linestyle='--', lw=2, alpha=0.7)
        ax2.set_xlabel('Predicted (mg/m3)', fontweight='bold')
        ax2.set_ylabel('Residuals', fontweight='bold')
        ax2.set_title('(B) Residual Plot', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Residual distribution (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(residuals, bins=40, alpha=0.7, color='forestgreen', edgecolor='black')
        ax3.axvline(x=0, color='r', linestyle='--', lw=2, alpha=0.7)
        ax3.set_xlabel('Residuals (mg/m3)', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('(C) Error Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Forecast horizon - R2 (bottom-left)
        ax4 = fig.add_subplot(gs[1, 0])
        timesteps = list(per_timestep_metrics.keys())
        r2_values = [per_timestep_metrics[t]['r2'] for t in timesteps]
        x_pos = range(len(timesteps))
        ax4.plot(x_pos, r2_values, 'o-', linewidth=2, markersize=8, color='steelblue')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(timesteps, rotation=45, ha='right')
        ax4.set_ylabel('R2', fontweight='bold')
        ax4.set_xlabel('Forecast Step', fontweight='bold')
        ax4.set_title('(D) R2 by Forecast Horizon', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1])
        
        # 5. Forecast horizon - RMSE (bottom-middle)
        ax5 = fig.add_subplot(gs[1, 1])
        rmse_values = [per_timestep_metrics[t]['rmse'] for t in timesteps]
        ax5.plot(x_pos, rmse_values, 'o-', linewidth=2, markersize=8, color='coral')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(timesteps, rotation=45, ha='right')
        ax5.set_ylabel('RMSE (mg/m3)', fontweight='bold')
        ax5.set_xlabel('Forecast Step', fontweight='bold')
        ax5.set_title('(E) RMSE by Forecast Horizon', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Metrics summary table (bottom-right)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        table_data = [
            ['Metric', 'Value'],
            ['R2', f"{metrics['r2']:.4f}"],
            ['RMSE', f"{metrics['rmse']:.4f} mg/m3"],
            ['MAE', f"{metrics['mae']:.4f} mg/m3"],
            ['MAPE', f"{metrics['mape']:.2f}%"],
            ['Bias', f"{metrics['bias']:.4f} mg/m3"],
            ['Samples', f"{metrics['n_samples']:,}"]
        ]
        
        table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax6.set_title('(F) Performance Summary', fontweight='bold', pad=20)
        
        plt.suptitle('Graph Neural Network Model Performance - Comprehensive Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        output_path = f"{self.output_dir}/{save_name}"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")
        return output_path
