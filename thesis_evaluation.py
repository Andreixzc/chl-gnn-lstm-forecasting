"""
Thesis Evaluation Module - Essential Metrics for Chlorophyll Prediction
Calculates standard performance metrics for model validation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats


class ThesisEvaluator:
    """Calculate comprehensive evaluation metrics for thesis presentation"""
    
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
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'bias': bias,
            'max_error': max_error,
            'correlation': corr,
            'p_value': p_value,
            'n_samples': len(y_true_clean)
        }
        
        return metrics
    
    def print_metrics_summary(self):
        """Print formatted metrics summary for thesis"""
        print("\n" + "="*70)
        print("THESIS EVALUATION RESULTS - MODEL PERFORMANCE METRICS")
        print("="*70)
        
        # Overall metrics
        print("\n📊 OVERALL PERFORMANCE (All Timesteps Combined)")
        print("-"*70)
        overall = self.metrics['overall']
        print(f"  R² (Coefficient of Determination):  {overall['r2']:.4f}")
        print(f"  RMSE (Root Mean Square Error):      {overall['rmse']:.4f} mg/m³")
        print(f"  MAE (Mean Absolute Error):          {overall['mae']:.4f} mg/m³")
        print(f"  MAPE (Mean Absolute % Error):       {overall['mape']:.2f}%")
        print(f"  Bias (Systematic Error):            {overall['bias']:.4f} mg/m³")
        print(f"  Maximum Error:                      {overall['max_error']:.4f} mg/m³")
        print(f"  Pearson Correlation:                {overall['correlation']:.4f}")
        print(f"  Number of Samples:                  {overall['n_samples']:,}")
        
        # Per-timestep metrics
        if self.per_timestep_metrics:
            print("\n📈 PER-TIMESTEP PERFORMANCE (Forecast Horizon Analysis)")
            print("-"*70)
            print(f"{'Timestep':<15} {'R²':<10} {'RMSE':<12} {'MAE':<12} {'MAPE':<10}")
            print("-"*70)
            
            for timestep, metrics in self.per_timestep_metrics.items():
                print(f"{timestep:<15} {metrics['r2']:<10.4f} "
                      f"{metrics['rmse']:<12.4f} {metrics['mae']:<12.4f} "
                      f"{metrics['mape']:<10.2f}%")
        
        print("="*70 + "\n")
    
    def export_latex_table(self, output_file='metrics_table.tex'):
        """Export metrics as LaTeX table for thesis document"""
        latex = []
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Model Performance Metrics}")
        latex.append("\\label{tab:model_metrics}")
        latex.append("\\begin{tabular}{lc}")
        latex.append("\\hline")
        latex.append("\\textbf{Metric} & \\textbf{Value} \\\\")
        latex.append("\\hline")
        
        overall = self.metrics['overall']
        latex.append(f"R² (Coefficient of Determination) & {overall['r2']:.4f} \\\\")
        latex.append(f"RMSE (mg/m³) & {overall['rmse']:.4f} \\\\")
        latex.append(f"MAE (mg/m³) & {overall['mae']:.4f} \\\\")
        latex.append(f"MAPE (\\%) & {overall['mape']:.2f} \\\\")
        latex.append(f"Bias (mg/m³) & {overall['bias']:.4f} \\\\")
        latex.append(f"Max Error (mg/m³) & {overall['max_error']:.4f} \\\\")
        latex.append(f"Pearson Correlation & {overall['correlation']:.4f} \\\\")
        latex.append(f"Sample Size & {overall['n_samples']:,} \\\\")
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        latex_str = "\n".join(latex)
        
        with open(output_file, 'w') as f:
            f.write(latex_str)
        
        print(f"✅ LaTeX table exported: {output_file}")
        return latex_str
    
    def export_per_timestep_latex(self, output_file='per_timestep_metrics.tex'):
        """Export per-timestep metrics as LaTeX table"""
        latex = []
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Performance Metrics by Forecast Horizon}")
        latex.append("\\label{tab:timestep_metrics}")
        latex.append("\\begin{tabular}{lcccc}")
        latex.append("\\hline")
        latex.append("\\textbf{Timestep} & \\textbf{R²} & \\textbf{RMSE} & \\textbf{MAE} & \\textbf{MAPE (\\%)} \\\\")
        latex.append("\\hline")
        
        for timestep, metrics in self.per_timestep_metrics.items():
            latex.append(f"{timestep} & {metrics['r2']:.4f} & "
                        f"{metrics['rmse']:.4f} & {metrics['mae']:.4f} & "
                        f"{metrics['mape']:.2f} \\\\")
        
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        latex_str = "\n".join(latex)
        
        with open(output_file, 'w') as f:
            f.write(latex_str)
        
        print(f"✅ Per-timestep LaTeX table exported: {output_file}")
        return latex_str
    
    def classify_water_quality(self, chl_values):
        """
        Classify water quality based on chlorophyll-a concentration
        
        Classification (typical for freshwater):
        - Oligotrophic: < 2.5 mg/m³
        - Mesotrophic: 2.5 - 8 mg/m³
        - Eutrophic: 8 - 25 mg/m³
        - Hypereutrophic: > 25 mg/m³
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
        
        print("\n🎯 WATER QUALITY CLASSIFICATION ACCURACY")
        print("-"*70)
        print(f"Overall Classification Accuracy: {accuracy*100:.2f}%")
        print("\nConfusion Matrix:")
        print(pd.DataFrame(cm, 
                          index=class_names, 
                          columns=class_names))
        print("\nDetailed Classification Report:")
        print(report)
        
        return {
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'report': report
        }
