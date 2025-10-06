#!/usr/bin/env python3
"""
Example usage of the Sports Betting ML System.

This script demonstrates how to use the various components of the system
for data collection, feature engineering, model training, and backtesting.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection.odds_api import OddsAPIClient
from data_collection.nba_api import NBAPlayerStats
from feature_engineering.feature_builder import SportsFeatureBuilder
from models.baseline_models import create_baseline_models, ModelEnsemble
from backtesting.backtester import Backtester, ConservativeStrategy, AggressiveStrategy
from config.config import get_config, ODDS_API_KEY


def example_data_collection():
    """Example of data collection from APIs."""
    print("=" * 60)
    print("EXAMPLE 1: DATA COLLECTION")
    print("=" * 60)
    
    # Initialize Odds API client
    odds_client = OddsAPIClient(ODDS_API_KEY)
    
    # Fetch NBA odds
    print("Fetching NBA odds...")
    nba_odds = odds_client.get_odds("basketball_nba")
    
    if nba_odds:
        df_odds = odds_client.flatten_odds_data(nba_odds)
        print(f"✓ Collected {len(df_odds)} odds records")
        print(f"  Sample data:")
        print(df_odds[['home_team', 'away_team', 'bookmaker', 'market']].head())
    else:
        print("✗ Failed to fetch odds data")
    
    # Initialize NBA API client
    print("\nFetching NBA player statistics...")
    nba_client = NBAPlayerStats()
    player_stats = nba_client.get_player_season_stats()
    
    if not player_stats.empty:
        print(f"✓ Collected stats for {len(player_stats)} players")
        print(f"  Sample data:")
        print(player_stats[['PLAYER_NAME', 'PTS', 'REB', 'AST']].head())
    else:
        print("✗ Failed to fetch player stats")
    
    return df_odds if nba_odds else None, player_stats


def example_feature_engineering():
    """Example of feature engineering."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: FEATURE ENGINEERING")
    print("=" * 60)
    
    # Create sample player data
    np.random.seed(42)
    n_games = 50
    
    sample_data = pd.DataFrame({
        'PLAYER_ID': ['player_001'] * n_games,
        'GAME_DATE': pd.date_range('2024-01-01', periods=n_games, freq='D'),
        'PTS': np.random.normal(20, 5, n_games),
        'REB': np.random.normal(8, 3, n_games),
        'AST': np.random.normal(6, 2, n_games),
        'STL': np.random.normal(1.5, 0.5, n_games),
        'BLK': np.random.normal(1, 0.5, n_games),
        'TOV': np.random.normal(3, 1, n_games),
        'MATCHUP': ['vs. Team A', '@ Team B'] * (n_games // 2)
    })
    
    print(f"Created sample data with {len(sample_data)} games")
    
    # Build features
    feature_builder = SportsFeatureBuilder()
    features = feature_builder.build_player_features(sample_data)
    
    print(f"✓ Built features with {len(features.columns)} columns")
    print(f"  New feature columns:")
    new_features = [col for col in features.columns if col not in sample_data.columns]
    print(f"  {new_features[:10]}...")  # Show first 10 new features
    
    return features


def example_model_training():
    """Example of model training."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: MODEL TRAINING")
    print("=" * 60)
    
    # Create sample training data
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'feature4': np.random.normal(0, 1, n_samples),
        'categorical': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    # Create target with some relationship to features
    y = pd.Series(
        2 * X['feature1'] + 1.5 * X['feature2'] - X['feature3'] + 
        np.random.normal(0, 0.5, n_samples)
    )
    
    print(f"Created training data: {len(X)} samples, {len(X.columns)} features")
    
    # Create and train models
    models = create_baseline_models()
    
    print("\nTraining models...")
    for name, model in models.items():
        print(f"  Training {name}...")
        metrics = model.train(X, y)
        print(f"    Validation R²: {metrics['val_r2']:.3f}")
    
    # Create ensemble
    print("\nTraining ensemble...")
    ensemble = ModelEnsemble([models['random_forest'], models['xgboost']])
    ensemble_metrics = ensemble.train(X, y)
    print(f"  Ensemble Validation R²: {ensemble_metrics['ensemble_val_r2']:.3f}")
    
    return models, ensemble


def example_backtesting():
    """Example of backtesting."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: BACKTESTING")
    print("=" * 60)
    
    # Create sample historical data
    np.random.seed(42)
    n_games = 500
    
    historical_data = pd.DataFrame({
        'game_id': [f'game_{i}' for i in range(n_games)],
        'date': pd.date_range('2024-01-01', periods=n_games, freq='D'),
        'home_team': np.random.choice(['Team A', 'Team B', 'Team C'], n_games),
        'away_team': np.random.choice(['Team X', 'Team Y', 'Team Z'], n_games)
    })
    
    predictions = pd.DataFrame({
        'game_id': [f'game_{i}' for i in range(n_games)],
        'predicted_value': np.random.normal(20, 5, n_games),
        'probability': np.random.uniform(0.4, 0.6, n_games),
        'edge': np.random.uniform(-0.1, 0.1, n_games)
    })
    
    odds_data = pd.DataFrame({
        'game_id': [f'game_{i}' for i in range(n_games)],
        'odds_decimal': np.random.uniform(1.5, 3.0, n_games),
        'odds_american': np.random.choice([-110, -105, +110, +120], n_games)
    })
    
    print(f"Created backtest data: {n_games} games")
    
    # Test different strategies
    strategies = [
        ConservativeStrategy(initial_bankroll=10000),
        AggressiveStrategy(initial_bankroll=10000)
    ]
    
    print("\nRunning backtests...")
    for strategy in strategies:
        print(f"  Testing {strategy.name} strategy...")
        backtester = Backtester(strategy)
        performance = backtester.run_backtest(historical_data, predictions, odds_data)
        
        print(f"    Final Bankroll: ${performance['final_bankroll']:,.2f}")
        print(f"    ROI: {performance['roi_percent']:.2f}%")
        print(f"    Win Rate: {performance['win_rate']:.2%}")
        print(f"    Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
    
    return strategies


def example_prediction():
    """Example of making predictions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: MAKING PREDICTIONS")
    print("=" * 60)
    
    # Create sample data for prediction
    np.random.seed(42)
    X_new = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 5),
        'feature2': np.random.normal(0, 1, 5),
        'feature3': np.random.normal(0, 1, 5),
        'feature4': np.random.normal(0, 1, 5),
        'categorical': ['A', 'B', 'C', 'A', 'B']
    })
    
    print("Sample data for prediction:")
    print(X_new)
    
    # Train a simple model
    models = create_baseline_models()
    X_train = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.normal(0, 1, 100),
        'feature4': np.random.normal(0, 1, 100),
        'categorical': np.random.choice(['A', 'B', 'C'], 100)
    })
    y_train = pd.Series(2 * X_train['feature1'] + 1.5 * X_train['feature2'] + np.random.normal(0, 0.5, 100))
    
    # Train model
    model = models['random_forest']
    model.train(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_new)
    
    print(f"\n✓ Generated {len(predictions)} predictions:")
    for i, pred in enumerate(predictions):
        print(f"  Sample {i+1}: {pred:.2f}")
    
    return predictions


def main():
    """Run all examples."""
    print("SPORTS BETTING ML SYSTEM - EXAMPLE USAGE")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run examples
        example_data_collection()
        example_feature_engineering()
        example_model_training()
        example_backtesting()
        example_prediction()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the main pipeline: python src/main.py")
        print("3. Explore the Jupyter notebooks in the notebooks/ directory")
        print("4. Customize the configuration in config/config.py")
        print("5. Add your own API keys and data sources")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
