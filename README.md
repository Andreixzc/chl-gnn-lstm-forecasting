# Spatial-Temporal Chlorophyll-a Forecasting using Graph Neural Networks

A proof-of-concept framework for reservoir water quality prediction combining Graph Neural Networks with satellite remote sensing. This project leverages the [Chl-CONNECT](https://github.com/manhtranduy/Chl-CONNECT) algorithm for chlorophyll-a estimation from Sentinel-2 imagery and extends it with deep learning-based spatial-temporal forecasting.

## Overview

This framework predicts chlorophyll-a concentration in freshwater reservoirs using:

- **Chl-CONNECT Algorithm**: Chlorophyll-a estimation from Sentinel-2 satellite imagery
- **Graph Neural Networks (GNN)**: Spatial relationship modeling between water pixels
- **LSTM Networks**: Temporal pattern learning for time series forecasting
- **High-Resolution Interpolation**: Smooth surface generation for visualization

Applied to the Três Marias Reservoir in Brazil using Google Earth Engine for data acquisition and PyTorch Geometric for graph-based deep learning.

## Architecture

### Data Pipeline

1. Grid-based sampling of Sentinel-2 imagery via Google Earth Engine
2. Chlorophyll-a estimation using Chl-CONNECT algorithm
3. Cloud and water masking with NDWI thresholding
4. Temporal interpolation for gap-filling
5. Spatial graph construction based on geographic proximity

### Model Structure

```
Input: Chlorophyll-a time series (T × N × F)
       T = sequence length, N = pixels, F = features

1. Graph Convolutional Layers
   ├─ GCN 1: F → 256 (ReLU + Dropout)
   └─ GCN 2: 256 → 128 (ReLU + Dropout)

2. LSTM Layer (hidden: 128)
   └─ Temporal dependency modeling

3. Fully Connected Layers
   ├─ FC 1: 128 → 64 (ReLU + Dropout)
   └─ FC 2: 64 → prediction_steps

Output: Multi-step future predictions (N × steps)
```

## Results

### Prediction Outputs

The framework generates high-resolution satellite overlay maps showing predicted chlorophyll-a concentrations:

**Step 5 Prediction**
![Satellite Overlay Step 5](assets/satellite_overlay_step_5.png)

**Step 6 Prediction**
![Satellite Overlay Step 6](assets/satellite_overlay_step_6.png)

### Performance Metrics

Model evaluation on validation set:

| Metric | Value |
|--------|-------|
| R² Score | 0.607 |
| RMSE | 9.71 mg/m³ |
| MAE | 7.29 mg/m³ |
| MAPE | 28.59% |
| Pearson Correlation | 0.804 |

## Usage

Extract satellite data and compute chlorophyll-a:
```bash
python Time_Series.py
```

Interpolate temporal gaps:
```bash
python interpolate_missing_5day_steps.py
```

Train model and generate predictions:
```bash
python graph_neural_network_daily_snapshots.py
```

## Framework Concept

This is a conceptual framework demonstrating integration of remote sensing algorithms with graph-based deep learning for water quality prediction. The approach is designed to be adaptable to other water bodies and parameters. Potential extensions include multi-parameter prediction, uncertainty quantification, and real-time forecasting pipelines

