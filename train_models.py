#!/usr/bin/env python3
"""
Train models on collected data.
This is a simplified training script to get started quickly.
"""

import sys
import pandas as pd
import numpy as np

sys.path.append('src')

from feature_engineering.feature_builder import SportsFeatureBuilder
from models.baseline_models import PlayerPropModel, create_baseline_models

def train_player_prop_models():
    """Train models to predict player props."""
    
    print("="*60)
    print("TRAINING PLAYER PROP MODELS")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    try:
        player_stats = pd.read_csv('data/player_stats.csv')
        game_logs = pd.read_csv('data/game_logs.csv')
        print(f"   ✓ Loaded {len(player_stats)} player records")
        print(f"   ✓ Loaded {len(game_logs)} game logs")
    except FileNotFoundError:
        print("   ❌ Data files not found!")
        print("   Run 'python3 collect_training_data.py' first.")
        return
    
    # Build features
    print("\n2. Building features...")
    feature_builder = SportsFeatureBuilder()
    
    # Use game logs for training (has historical data)
    features = feature_builder.build_player_features(game_logs)
    print(f"   ✓ Built {len(features.columns)} features")
    
    # Prepare training data
    print("\n3. Preparing training data...")
    
    # Select numeric features only
    X = features.select_dtypes(include=[np.number]).copy()
    
    # Remove target variables if they exist
    target_cols = ['PTS', 'REB', 'AST']
    available_targets = [col for col in target_cols if col in X.columns]
    
    if not available_targets:
        print("   ❌ No target columns found in data!")
        return
    
    # Train a model for each stat type
    models_trained = {}
    
    for target in available_targets:
        print(f"\n4. Training model for {target}...")
        
        # Prepare features and target
        y = X[target].copy()
        X_train = X.drop(columns=target_cols, errors='ignore')
        
        # Remove rows with missing values
        valid_rows = ~(X_train.isna().any(axis=1) | y.isna())
        X_train = X_train[valid_rows]
        y_train = y[valid_rows]
        
        if len(X_train) < 50:
            print(f"   ⚠ Not enough data for {target} ({len(X_train)} samples)")
            continue
        
        print(f"   Training on {len(X_train)} samples...")
        
        # Create and train model
        model = PlayerPropModel(model_name=f"{target}_Model")
        
        try:
            metrics = model.train(X_train, y_train)
            
            print(f"   ✓ Model trained!")
            print(f"     - Validation R²: {metrics['val_r2']:.3f}")
            print(f"     - Validation MAE: {metrics['val_mae']:.2f}")
            
            # Save model
            model.save_model(f'models/{target.lower()}_model.pkl')
            print(f"   ✓ Saved to models/{target.lower()}_model.pkl")
            
            models_trained[target] = {
                'model': model,
                'r2': metrics['val_r2'],
                'mae': metrics['val_mae']
            }
            
        except Exception as e:
            print(f"   ❌ Error training {target} model: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    if models_trained:
        print(f"\n✓ Successfully trained {len(models_trained)} models:")
        for stat, info in models_trained.items():
            print(f"  - {stat}: R² = {info['r2']:.3f}, MAE = {info['mae']:.2f}")
        
        print(f"\n🎉 Models ready for predictions!")
        print(f"   Next: Run 'python3 make_predictions.py'")
    else:
        print("\n❌ No models were trained successfully.")
        print("   Check data quality and try again.")
    
    return models_trained


if __name__ == "__main__":
    try:
        models = train_player_prop_models()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

