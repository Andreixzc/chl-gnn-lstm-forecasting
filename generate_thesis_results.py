"""
GENERATE THESIS RESULTS - Quick Runner Script
Run this to generate all metrics, visualizations, and results for your thesis

This script will:
1. Train the Graph Neural Network model (if not already trained)
2. Evaluate performance on validation data
3. Calculate all thesis metrics (R², RMSE, MAE, MAPE, etc.)
4. Generate publication-quality visualizations
5. Export LaTeX tables for your thesis document
6. Create future chlorophyll predictions

Estimated runtime: 10-20 minutes (depending on your hardware)
"""

import os
import sys
from datetime import datetime

def main():
    print("\n" + "="*70)
    print("THESIS RESULTS GENERATOR")
    print("Chlorophyll-a Prediction using Graph Neural Networks")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Check if data exists
    if not os.path.exists('daily_snapshots'):
        print("❌ ERROR: 'daily_snapshots' directory not found!")
        print("   Please run Time_Series.py first to generate the data.")
        return
    
    # Count snapshot files
    import glob
    snapshots = glob.glob('daily_snapshots/snapshot_*.csv')
    print(f"✅ Found {len(snapshots)} daily snapshot files")
    
    if len(snapshots) == 0:
        print("❌ ERROR: No snapshot files found in daily_snapshots/")
        print("   Please run Time_Series.py first to generate the data.")
        return
    
    # Import main forecaster
    print("\n📦 Loading Graph Neural Network forecaster...")
    from graph_neural_network_daily_snapshots import GraphChlorophyllForecaster
    
    # Initialize forecaster
    print("🔧 Initializing forecaster...")
    forecaster = GraphChlorophyllForecaster(
        data_dir="daily_snapshots",
        aoi_path="Area.json"
    )
    
    # Run complete analysis with thesis evaluation
    print("\n🚀 Starting complete analysis with thesis evaluation...")
    print("   This will take 10-20 minutes. Please be patient...\n")
    
    try:
        forecaster.run_complete_analysis(include_thesis_evaluation=True)
        
        # Success summary
        print("\n" + "="*70)
        print("✅ THESIS RESULTS GENERATION COMPLETE!")
        print("="*70)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n📂 YOUR THESIS FILES ARE READY:")
        print("\n1️⃣  METRICS (copy these into your thesis):")
        print("    📄 thesis_results/metrics_table.tex")
        print("    📄 thesis_results/per_timestep_metrics.tex")
        
        print("\n2️⃣  FIGURES (insert these in your thesis document):")
        print("    🖼️  thesis_figures/comprehensive_summary.png (USE THIS ONE!)")
        print("    🖼️  thesis_figures/predicted_vs_actual.png")
        print("    🖼️  thesis_figures/residual_analysis.png")
        print("    🖼️  thesis_figures/forecast_horizon_performance.png")
        print("    🖼️  thesis_figures/training_history.png")
        
        print("\n3️⃣  FUTURE PREDICTIONS:")
        print("    📄 predictions_csv/all_predictions_combined.csv")
        print("    🗺️  satellite_maps/ (overlay maps)")
        
        print("\n4️⃣  TRAINED MODEL:")
        print("    💾 best_graph_model.pth")
        
        print("\n" + "="*70)
        print("📝 NEXT STEPS FOR YOUR THESIS:")
        print("="*70)
        print("1. Open thesis_figures/comprehensive_summary.png")
        print("   → This is your main results figure!")
        print("\n2. Copy thesis_results/metrics_table.tex into your LaTeX document")
        print("   → These are your performance metrics")
        print("\n3. Use the satellite maps in satellite_maps/ to show predictions")
        print("\n4. The metrics show your model's accuracy - use them in your")
        print("   results and discussion sections!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        print("\nIf you see CUDA/GPU errors, the model will use CPU (slower but will work)")
        return

if __name__ == "__main__":
    main()
