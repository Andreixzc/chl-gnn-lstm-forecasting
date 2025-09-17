"""
Enhanced CHL-CONNECT Forecasting with Spectral Features
Uses all available spectral bands for improved predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EnhancedChlConnectForecaster:
    """Enhanced forecasting using spectral features"""
    
    def __init__(self, data_path="chl_connect_timeseries_2000pts.csv"):
        self.data_path = data_path
        self.data = None
        self.scaler = StandardScaler()
        self.model = None
        
    def load_and_prepare_data(self):
        """Load and prepare data with all spectral features"""
        print("📊 Loading CHL-CONNECT data with spectral features...")
        
        self.data = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(self.data)} observations")
        
        # Convert date
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # Add temporal features
        self.data['day_of_year'] = self.data['date'].dt.dayofyear
        self.data['month'] = self.data['date'].dt.month
        self.data['season'] = ((self.data['month'] + 2) // 3) % 4 + 1
        self.data['year'] = self.data['date'].dt.year
        
        # Calculate spectral indices
        self.data['ndci'] = (self.data['Rrs705'] - self.data['Rrs665']) / (self.data['Rrs705'] + self.data['Rrs665'])
        self.data['ndvi'] = (self.data['Rrs865'] - self.data['Rrs665']) / (self.data['Rrs865'] + self.data['Rrs665'])
        self.data['fai'] = self.data['Rrs865'] - (self.data['Rrs665'] + (self.data['Rrs865'] - self.data['Rrs665']) * 0.4)
        
        # Band ratios
        self.data['ratio_705_665'] = self.data['Rrs705'] / self.data['Rrs665']
        self.data['ratio_740_705'] = self.data['Rrs740'] / self.data['Rrs705']
        self.data['ratio_560_490'] = self.data['Rrs560'] / self.data['Rrs490']
        
        print(f"✅ Enhanced data prepared with spectral indices")
        return self.data
    
    def train_enhanced_model(self):
        """Train model using all spectral and temporal features"""
        if self.data is None:
            self.load_and_prepare_data()
            
        print("🤖 Training enhanced spectral model...")
        
        # Feature columns (all spectral bands + indices + temporal)
        feature_cols = [
            'Rrs443', 'Rrs490', 'Rrs560', 'Rrs665', 'Rrs705', 'Rrs740', 'Rrs783', 'Rrs865',
            'ndci', 'ndvi', 'fai',
            'ratio_705_665', 'ratio_740_705', 'ratio_560_490',
            'day_of_year', 'month', 'season'
        ]
        
        # Prepare features and target
        X = self.data[feature_cols].values
        y = self.data['chlorophyll_a'].values
        
        # Remove NaN and infinite values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(X).any(axis=1) | np.isinf(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        print(f"   Training on {len(X_clean)} samples with {len(feature_cols)} features")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Train Random Forest
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
        self.model.fit(X_scaled, y_clean)
        
        # Feature importance
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("✅ Model trained. Top 5 most important features:")
        for _, row in feature_importance.head().iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        # Model performance
        score = self.model.score(X_scaled, y_clean)
        print(f"   Model R²: {score:.3f}")
        
        return self.model
    
    def generate_future_predictions_enhanced(self, months_ahead=6):
        """Generate enhanced future predictions"""
        if self.model is None:
            self.train_enhanced_model()
            
        print(f"🔮 Generating enhanced predictions for {months_ahead} months...")
        
        # Generate future dates
        last_date = self.data['date'].max()
        future_dates = []
        for i in range(1, months_ahead + 1):
            future_date = last_date + timedelta(days=30*i)
            future_dates.append(future_date)
        
        # Get unique pixels with their recent spectral characteristics
        unique_pixels = self.data.groupby(['lon', 'lat']).last().reset_index()
        
        all_predictions = []
        
        for idx, pixel in unique_pixels.iterrows():
            if (idx + 1) % 10 == 0:
                print(f"   Processing pixel {idx + 1}/{len(unique_pixels)}...")
            
            for future_date in future_dates:
                # Use latest spectral values as baseline
                base_features = pixel.copy()
                
                # Update temporal features
                base_features['day_of_year'] = future_date.timetuple().tm_yday
                base_features['month'] = future_date.month
                base_features['season'] = ((future_date.month + 2) // 3) % 4 + 1
                
                # Apply seasonal adjustments to spectral bands
                seasonal_factor = {1: 1.1, 2: 1.0, 3: 0.9, 4: 0.85, 5: 0.8, 6: 0.75,
                                 7: 0.8, 8: 0.9, 9: 1.0, 10: 1.1, 11: 1.2, 12: 1.15}
                factor = seasonal_factor.get(future_date.month, 1.0)
                
                # Adjust spectral bands
                spectral_bands = ['Rrs443', 'Rrs490', 'Rrs560', 'Rrs665', 'Rrs705', 'Rrs740', 'Rrs783', 'Rrs865']
                for band in spectral_bands:
                    base_features[band] *= factor
                
                # Recalculate indices with adjusted bands
                base_features['ndci'] = (base_features['Rrs705'] - base_features['Rrs665']) / (base_features['Rrs705'] + base_features['Rrs665'])
                base_features['ndvi'] = (base_features['Rrs865'] - base_features['Rrs665']) / (base_features['Rrs865'] + base_features['Rrs665'])
                base_features['fai'] = base_features['Rrs865'] - (base_features['Rrs665'] + (base_features['Rrs865'] - base_features['Rrs665']) * 0.4)
                base_features['ratio_705_665'] = base_features['Rrs705'] / base_features['Rrs665']
                base_features['ratio_740_705'] = base_features['Rrs740'] / base_features['Rrs705']
                base_features['ratio_560_490'] = base_features['Rrs560'] / base_features['Rrs490']
                
                # Prepare feature vector
                feature_cols = [
                    'Rrs443', 'Rrs490', 'Rrs560', 'Rrs665', 'Rrs705', 'Rrs740', 'Rrs783', 'Rrs865',
                    'ndci', 'ndvi', 'fai',
                    'ratio_705_665', 'ratio_740_705', 'ratio_560_490',
                    'day_of_year', 'month', 'season'
                ]
                
                features = base_features[feature_cols].values.reshape(1, -1)
                
                # Handle any NaN values
                try:
                    if np.isnan(features.astype(float)).any():
                        continue
                except (TypeError, ValueError):
                    continue
                
                # Scale and predict
                features_scaled = self.scaler.transform(features)
                prediction = self.model.predict(features_scaled)[0]
                
                # Ensure realistic bounds
                prediction = np.clip(prediction, 0.5, 200.0)
                
                all_predictions.append({
                    'pixel_x': pixel['lon'],
                    'pixel_y': pixel['lat'],
                    'date': future_date.strftime('%Y-%m-%d'),
                    'chlorophyll': prediction,
                    'prediction_type': 'enhanced_spectral_forecast'
                })
        
        predictions_df = pd.DataFrame(all_predictions)
        print(f"✅ Generated {len(predictions_df)} enhanced predictions")
        
        return predictions_df
    
    def create_enhanced_intensity_maps(self, predictions_df):
        """Create enhanced intensity maps"""
        print("🗺️ Creating enhanced intensity maps...")
        
        # Save predictions
        predictions_df.to_csv('enhanced_chl_connect_predictions.csv', index=False)
        
        # Get all unique dates
        unique_dates = sorted(predictions_df['date'].unique())
        
        # Create comprehensive visualization
        n_dates = len(unique_dates)
        cols = 3
        rows = (n_dates + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Enhanced Future Chlorophyll-a Intensity Maps\n(Spectral-Based Forecasting)', fontsize=16, fontweight='bold')
        
        for i, date in enumerate(unique_dates):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            date_data = predictions_df[predictions_df['date'] == date]
            
            # Create scatter plot
            scatter = ax.scatter(
                date_data['pixel_x'], 
                date_data['pixel_y'], 
                c=date_data['chlorophyll'],
                cmap='RdYlGn_r',
                s=30,
                alpha=0.8,
                vmin=0,
                vmax=predictions_df['chlorophyll'].quantile(0.95)  # Dynamic scale
            )
            
            ax.set_title(f'{date}\n({len(date_data)} pixels)')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Chlorophyll-a (mg/m³)')
            
            # Add statistics
            stats_text = f"""Mean: {date_data['chlorophyll'].mean():.1f}
Range: {date_data['chlorophyll'].min():.1f}-{date_data['chlorophyll'].max():.1f}"""
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for i in range(n_dates, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('enhanced_chl_connect_intensity_maps.png', dpi=300, bbox_inches='tight')
        print("📊 Enhanced intensity maps saved to: enhanced_chl_connect_intensity_maps.png")
        plt.show()
        
        return predictions_df
    
    def run_enhanced_pipeline(self):
        """Run complete enhanced pipeline"""
        print("🚀 ENHANCED CHL-CONNECT FORECASTING PIPELINE")
        print("=" * 60)
        
        # Load data with spectral features
        self.load_and_prepare_data()
        
        # Train enhanced model
        self.train_enhanced_model()
        
        # Generate enhanced predictions
        predictions = self.generate_future_predictions_enhanced(months_ahead=6)
        
        # Create enhanced maps
        self.create_enhanced_intensity_maps(predictions)
        
        print(f"\n🎉 ENHANCED FORECASTING COMPLETE!")
        print(f"📊 Generated {len(predictions)} spectral-based predictions")
        print(f"🌊 Chlorophyll range: {predictions['chlorophyll'].min():.1f} - {predictions['chlorophyll'].max():.1f} mg/m³")
        
        return predictions

def main():
    """Main execution"""
    forecaster = EnhancedChlConnectForecaster("chl_connect_timeseries_2000pts.csv")
    enhanced_predictions = forecaster.run_enhanced_pipeline()
    return enhanced_predictions

if __name__ == "__main__":
    results = main()