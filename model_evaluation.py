"""
Model Evaluation Module - Performance Metrics for Chlorophyll Prediction
Calculates standard performance metrics for model validation and saves to JSON.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats


class ModelEvaluator:
    """Calculate comprehensive evaluation metrics for model performance"""
    
    def __init__(self):
        self.metrics = {}
        self.per_timestep_metrics = {}
        
    def calculate_all_metrics(self, y_true, y_pred, timestep_names=None):
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: Ground truth values, shape (n_samples, n_timesteps) or (n_samples,)
            y_pred: Predicted values, same shape as y_true
            timestep_names: Optional list of names for each timestep
            
        Returns:
            dict: Dictionary containing all metrics
        """
        # Handle both 1D and 2D arrays
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)
        
        n_timesteps = y_true.shape[1]
        
        # Overall metrics (all timesteps combined)
        self.metrics['overall'] = self._calculate_metrics_single(
            y_true.flatten(), 
            y_pred.flatten()
        )
        
        # Per-timestep metrics
        if timestep_names is None:
            timestep_names = [f"Step {i+1}" for i in range(n_timesteps)]
        
        for i, name in enumerate(timestep_names):
            self.per_timestep_metrics[name] = self._calculate_metrics_single(
                y_true[:, i],
                y_pred[:, i]
            )
        
        return self.metrics, self.per_timestep_metrics
    
    def _calculate_metrics_single(self, y_true, y_pred):
        """Calculate metrics for a single set of predictions"""
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {
                'r2': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'bias': np.nan,
                'max_error': np.nan,
                'n_samples': 0
            }
        
        # Calculate metrics
        r2 = r2_score(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        
        # MAPE (avoiding division by zero)
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-8))) * 100
        
        # Bias (systematic over/under prediction)
        bias = np.mean(y_pred_clean - y_true_clean)
        
        # Maximum error
        max_error = np.max(np.abs(y_true_clean - y_pred_clean))
        
        # Correlation coefficient
        corr, p_value = stats.pearsonr(y_true_clean, y_pred_clean)
        
        metrics = {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'bias': float(bias),
            'max_error': float(max_error),
            'correlation': float(corr),
            'p_value': float(p_value),
            'n_samples': int(len(y_true_clean))
        }
        
        return metrics
    
    def print_metrics_summary(self):
        """Print formatted metrics summary"""
        print("\n" + "="*70)
        print("MODEL EVALUATION RESULTS - PERFORMANCE METRICS")
        print("="*70)
        
        # Overall metrics
        print("\nOVERALL PERFORMANCE (All Timesteps Combined)")
        print("-"*70)
        overall = self.metrics['overall']
        print(f"  R2 (Coefficient of Determination):  {overall['r2']:.4f}")
        print(f"  RMSE (Root Mean Square Error):      {overall['rmse']:.4f} mg/m3")
        print(f"  MAE (Mean Absolute Error):          {overall['mae']:.4f} mg/m3")
        print(f"  MAPE (Mean Absolute % Error):       {overall['mape']:.2f}%")
        print(f"  Bias (Systematic Error):            {overall['bias']:.4f} mg/m3")
        print(f"  Maximum Error:                      {overall['max_error']:.4f} mg/m3")
        print(f"  Pearson Correlation:                {overall['correlation']:.4f}")
        print(f"  Number of Samples:                  {overall['n_samples']:,}")
        
        # Per-timestep metrics
        if self.per_timestep_metrics:
            print("\nPER-TIMESTEP PERFORMANCE (Forecast Horizon Analysis)")
            print("-"*70)
            print(f"{'Timestep':<15} {'R2':<10} {'RMSE':<12} {'MAE':<12} {'MAPE':<10}")
            print("-"*70)
            
            for timestep, metrics in self.per_timestep_metrics.items():
                print(f"{timestep:<15} {metrics['r2']:<10.4f} "
                      f"{metrics['rmse']:<12.4f} {metrics['mae']:<12.4f} "
                      f"{metrics['mape']:<10.2f}%")
        
        print("="*70 + "\n")
    
    def save_to_json(self, output_path='evaluation_results/metrics.json'):
        """
        Save all metrics to JSON file with timestamp
        
        Args:
            output_path: Path to save the JSON file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall': self.metrics.get('overall', {}),
            'per_timestep': self.per_timestep_metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Metrics saved to: {output_path}")
        return output_path
    
    def classify_water_quality(self, chl_values):
        """
        Classify water quality based on chlorophyll-a concentration
        
        Classification (typical for freshwater):
        - Oligotrophic: < 2.5 mg/m3
        - Mesotrophic: 2.5 - 8 mg/m3
        - Eutrophic: 8 - 25 mg/m3
        - Hypereutrophic: > 25 mg/m3
        """
        classes = np.zeros_like(chl_values, dtype=int)
        classes[chl_values < 2.5] = 0  # Oligotrophic
        classes[(chl_values >= 2.5) & (chl_values < 8)] = 1  # Mesotrophic
        classes[(chl_values >= 8) & (chl_values < 25)] = 2  # Eutrophic
        classes[chl_values >= 25] = 3  # Hypereutrophic
        
        return classes
    
    def evaluate_water_quality_classification(self, y_true, y_pred):
        """Calculate classification accuracy for water quality categories"""
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Classify
        true_classes = self.classify_water_quality(y_true.flatten())
        pred_classes = self.classify_water_quality(y_pred.flatten())
        
        # Remove NaN
        mask = ~(np.isnan(y_true.flatten()) | np.isnan(y_pred.flatten()))
        true_classes_clean = true_classes[mask]
        pred_classes_clean = pred_classes[mask]
        
        # Confusion matrix
        cm = confusion_matrix(true_classes_clean, pred_classes_clean)
        
        # Classification report
        class_names = ['Oligotrophic', 'Mesotrophic', 'Eutrophic', 'Hypereutrophic']
        report = classification_report(
            true_classes_clean, 
            pred_classes_clean,
            target_names=class_names,
            zero_division=0
        )
        
        # Overall accuracy
        accuracy = np.mean(true_classes_clean == pred_classes_clean)
        
        print("\nWATER QUALITY CLASSIFICATION ACCURACY")
        print("-"*70)
        print(f"Overall Classification Accuracy: {accuracy*100:.2f}%")
        print("\nConfusion Matrix:")
        print(pd.DataFrame(cm, 
                          index=class_names, 
                          columns=class_names))
        print("\nDetailed Classification Report:")
        print(report)
        
        return {
            'confusion_matrix': cm.tolist(),
            'accuracy': float(accuracy),
            'report': report
        }
