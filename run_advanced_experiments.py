#!/usr/bin/env python3
"""
Advanced Semi-Supervised Methods Experiment Runner
==================================================
Script to run FlexMatch-lite and Label Spreading experiments
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import warnings
from datetime import datetime
import zipfile
from io import StringIO
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from semi_supervised_library import (
    SemiDataConfig, FlexMatchConfig, LabelSpreadingConfig,
    run_flexmatch, run_label_spreading,
    mask_labels_time_aware, AQI_CLASSES
)

def load_and_prepare_data():
    """Load raw data and create features for advanced experiments"""
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    # Check if processed features exist
    features_file = processed_dir / "features_with_aqi.csv"
    if features_file.exists():
        print("📊 Loading existing processed data...")
        df = pd.read_csv(features_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    
    # Load from raw ZIP file
    zip_file = raw_dir / "PRSA2017_Data_20130301-20170228.zip"
    if not zip_file.exists():
        raise FileNotFoundError(f"Raw data file not found: {zip_file}")
    
    print("📦 Loading raw data from ZIP file...")
    
    # Read all CSV files from ZIP
    all_data = []
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for filename in zip_ref.namelist():
            if filename.endswith('.csv') and not filename.startswith('__'):
                print(f"   Reading {filename}...")
                with zip_ref.open(filename) as csvfile:
                    content = csvfile.read().decode('utf-8')
                    station_df = pd.read_csv(StringIO(content))
                    
                    # Extract station name from filename
                    station_name = filename.split('_')[-1].replace('.csv', '')
                    station_df['station'] = station_name
                    all_data.append(station_df)
    
    # Combine all stations
    df = pd.concat(all_data, ignore_index=True)
    
    # Basic preprocessing
    print("🔧 Basic preprocessing...")
    
    # Create datetime column
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    
    # Handle missing values
    numeric_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    
    # Create AQI classes based on PM2.5
    print("🏷️ Creating AQI classes...")
    def pm25_to_aqi_class(pm25):
        if pd.isna(pm25):
            return 'Moderate'  # Default for missing values
        elif pm25 <= 12:
            return 'Good'
        elif pm25 <= 35.4:
            return 'Moderate'
        elif pm25 <= 55.4:
            return 'Unhealthy_for_Sensitive_Groups'
        elif pm25 <= 150.4:
            return 'Unhealthy'
        elif pm25 <= 250.4:
            return 'Very_Unhealthy'
        else:
            return 'Hazardous'
    
    df['aqi_class'] = df['PM2.5'].apply(pm25_to_aqi_class)
    
    # Create time features
    print("⏰ Creating time features...")
    df['dow'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Create lag features (simplified)
    print("🔄 Creating lag features...")
    df = df.sort_values(['station', 'datetime'])
    
    for col in ['TEMP', 'PRES', 'DEWP', 'WSPM']:
        if col in df.columns:
            # Create lag features by station
            df[f'{col}_lag_1h'] = df.groupby('station')[col].shift(1)
            df[f'{col}_lag_6h'] = df.groupby('station')[col].shift(6)
            df[f'{col}_lag_24h'] = df.groupby('station')[col].shift(24)
            
            # Create rolling features
            df[f'{col}_rolling_6h'] = df.groupby('station')[col].transform(
                lambda x: x.rolling(window=6, min_periods=1).mean()
            )
    
    # Fill remaining NaN values
    df = df.ffill().bfill().fillna(0)
    
    # Save processed data
    print(f"💾 Saving processed data to {features_file}...")
    df.to_csv(features_file, index=False)
    
    print(f"   ✅ Dataset shape: {df.shape}")
    print(f"   ✅ Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"   ✅ Stations: {df['station'].nunique()}")
    print(f"   ✅ AQI distribution: {df['aqi_class'].value_counts().to_dict()}")
    
    return df

def main():
    print("🚀 Advanced Semi-Supervised Methods Experiment")
    print("=" * 60)
    
    # Setup paths
    data_dir = project_root / "data" / "processed"
    results_dir = data_dir / "advanced_semi_results"
    results_dir.mkdir(exist_ok=True)
    
    # Load data
    print("📊 Loading and preparing dataset...")
    try:
        df = load_and_prepare_data()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print(f"   Dataset shape: {df.shape}")
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"   AQI classes: {df['aqi_class'].value_counts().to_dict()}")
    
    # Setup configurations
    print("\n⚙️ Setting up configurations...")
    
    data_cfg = SemiDataConfig(
        target_col="aqi_class",
        cutoff="2017-01-01",
        random_state=42,
    )
    
    flexmatch_cfg = FlexMatchConfig(
        tau_base=0.60,  # Lower base threshold
        max_iter=10,
        min_new_per_iter=20,
        focal_alpha=0.25,
        focal_gamma=2.0,
        threshold_warmup=3
    )
    
    label_spreading_cfg = LabelSpreadingConfig(
        kernel="rbf",
        gamma=20,
        alpha=0.2,
        max_iter=30,
        n_neighbors=7
    )
    
    # Create semi-supervised dataset
    print("\n🎭 Creating semi-supervised dataset (95% unlabeled)...")
    
    # Sample dataset for testing (too large for Label Spreading)
    print("   📉 Sampling dataset to 10,000 samples for testing...")
    df_sampled = df.sample(n=10000, random_state=42).copy()
    
    df_semi = mask_labels_time_aware(df_sampled, cfg=data_cfg, missing_fraction=0.95)
    
    train_mask = df_semi['datetime'] < pd.Timestamp(data_cfg.cutoff)
    labeled_mask = df_semi['is_labeled']
    
    print(f"   Training samples: {train_mask.sum():,}")
    print(f"   Labeled training: {(train_mask & labeled_mask).sum():,}")
    print(f"   Unlabeled training: {(train_mask & ~labeled_mask).sum():,}")
    
    # Run FlexMatch-lite
    print("\n🔄 Running FlexMatch-lite...")
    print("   Features: Dynamic thresholds + Focal loss")
    
    try:
        flexmatch_results = run_flexmatch(df_semi, data_cfg, flexmatch_cfg)
        
        print(f"   ✅ Accuracy: {flexmatch_results['test_metrics']['accuracy']:.4f}")
        print(f"   ✅ F1-macro: {flexmatch_results['test_metrics']['f1_macro']:.4f}")
        
        # Save FlexMatch results
        flexmatch_save = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'tau_base': flexmatch_cfg.tau_base,
                'max_iter': flexmatch_cfg.max_iter,
                'min_new_per_iter': flexmatch_cfg.min_new_per_iter,
                'focal_alpha': flexmatch_cfg.focal_alpha,
                'focal_gamma': flexmatch_cfg.focal_gamma,
                'threshold_warmup': flexmatch_cfg.threshold_warmup
            },
            'test_metrics': flexmatch_results['test_metrics'],
            'history': flexmatch_results['history']
        }
        
        with open(results_dir / "flexmatch_results.json", 'w') as f:
            json.dump(flexmatch_save, f, indent=2, default=str)
            
    except Exception as e:
        print(f"   ❌ FlexMatch failed: {e}")
        flexmatch_results = None
    
    # Run Label Spreading
    print("\n🌐 Running Label Spreading...")
    print("   Features: Graph-based propagation + Global optimization")
    
    try:
        label_spreading_results = run_label_spreading(df_semi, data_cfg, label_spreading_cfg)
        
        print(f"   ✅ Accuracy: {label_spreading_results['test_metrics']['accuracy']:.4f}")
        print(f"   ✅ F1-macro: {label_spreading_results['test_metrics']['f1_macro']:.4f}")
        
        # Save Label Spreading results
        label_spreading_save = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'kernel': label_spreading_cfg.kernel,
                'gamma': label_spreading_cfg.gamma,
                'alpha': label_spreading_cfg.alpha,
                'max_iter': label_spreading_cfg.max_iter,
                'n_neighbors': label_spreading_cfg.n_neighbors
            },
            'test_metrics': label_spreading_results['test_metrics'],
            'history': label_spreading_results['history']
        }
        
        with open(results_dir / "label_spreading_results.json", 'w') as f:
            json.dump(label_spreading_save, f, indent=2, default=str)
            
    except Exception as e:
        print(f"   ❌ Label Spreading failed: {e}")
        label_spreading_results = None
    
    # Create comparison
    print("\n📊 Creating method comparison...")
    
    comparison_results = []
    
    # Add advanced results
    if flexmatch_results:
        comparison_results.append({
            'Method': 'FlexMatch-lite',
            'Type': 'Advanced Semi-Supervised',
            'Accuracy': flexmatch_results['test_metrics']['accuracy'],
            'F1-Macro': flexmatch_results['test_metrics']['f1_macro'],
            'Key Feature': 'Dynamic thresholds + Focal loss'
        })
    
    if label_spreading_results:
        comparison_results.append({
            'Method': 'Label Spreading',
            'Type': 'Advanced Semi-Supervised',
            'Accuracy': label_spreading_results['test_metrics']['accuracy'],
            'F1-Macro': label_spreading_results['test_metrics']['f1_macro'],
            'Key Feature': 'Graph-based propagation'
        })
    
    # Try to load baseline results for comparison
    try:
        baseline_files = [
            ("baseline_results.json", "Supervised Baseline", "Supervised", "Traditional supervised learning"),
            ("self_training_results.json", "Self-Training", "Basic Semi-Supervised", "Fixed threshold pseudo-labeling"),
            ("co_training_results.json", "Co-Training", "Basic Semi-Supervised", "Two-view collaboration")
        ]
        
        for filename, method_name, method_type, key_feature in baseline_files:
            filepath = data_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    baseline_data = json.load(f)
                
                comparison_results.append({
                    'Method': method_name,
                    'Type': method_type,
                    'Accuracy': baseline_data['test_metrics']['accuracy'],
                    'F1-Macro': baseline_data['test_metrics']['f1_macro'],
                    'Key Feature': key_feature
                })
                print(f"   ✅ Loaded {method_name} results")
            else:
                print(f"   ⚠️  {filename} not found - skipping baseline comparison")
                
    except Exception as e:
        print(f"   ⚠️  Error loading baseline results: {e}")
    
    # Save comparison
    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('F1-Macro', ascending=False)
        comparison_df.to_csv(results_dir / "method_comparison.csv", index=False)
        
        print("\n🏆 FINAL RESULTS SUMMARY")
        print("=" * 50)
        for _, row in comparison_df.iterrows():
            print(f"{row['Method']:<20} | Acc: {row['Accuracy']:.4f} | F1: {row['F1-Macro']:.4f}")
        print("=" * 50)
        
        # Highlight improvements
        advanced_methods = comparison_df[comparison_df['Type'] == 'Advanced Semi-Supervised']
        basic_methods = comparison_df[comparison_df['Type'] == 'Basic Semi-Supervised']
        
        if len(advanced_methods) > 0 and len(basic_methods) > 0:
            best_advanced = advanced_methods.loc[advanced_methods['F1-Macro'].idxmax()]
            best_basic = basic_methods.loc[basic_methods['F1-Macro'].idxmax()]
            
            improvement = best_advanced['F1-Macro'] - best_basic['F1-Macro']
            improvement_pct = (improvement / best_basic['F1-Macro']) * 100
            
            print(f"\n🚀 ADVANCED METHOD IMPROVEMENT:")
            print(f"   Best Advanced: {best_advanced['Method']} (F1: {best_advanced['F1-Macro']:.4f})")
            print(f"   Best Basic: {best_basic['Method']} (F1: {best_basic['F1-Macro']:.4f})")
            print(f"   Improvement: +{improvement:.4f} (+{improvement_pct:.1f}%)")
    
    print(f"\n💾 Results saved to: {results_dir}")
    print("📊 View results in dashboard: pages/4_Advanced_Methods.py")
    print("✅ Advanced semi-supervised experiment completed!")

if __name__ == "__main__":
    main()