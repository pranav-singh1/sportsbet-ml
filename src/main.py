"""
Main pipeline for the Sports Betting ML System.
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collection.odds_api import OddsAPIClient
from data_collection.nba_api import NBAPlayerStats
from feature_engineering.feature_builder import SportsFeatureBuilder
from models.baseline_models import create_baseline_models, ModelEnsemble
from backtesting.backtester import Backtester, ConservativeStrategy, AggressiveStrategy
from config.config import get_config, validate_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SportsBettingPipeline:
    """Main pipeline for the sports betting ML system."""
    
    def __init__(self):
        self.odds_client = None
        self.nba_client = None
        self.feature_builder = SportsFeatureBuilder()
        self.models = {}
        self.results = {}
        
        # Validate configuration
        issues = validate_config()
        if issues:
            logger.warning("Configuration issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
    
    def initialize_data_clients(self):
        """Initialize data collection clients."""
        logger.info("Initializing data collection clients...")
        
        # Initialize Odds API client
        from config.config import ODDS_API_KEY
        self.odds_client = OddsAPIClient(ODDS_API_KEY)
        
        # Initialize NBA API client
        self.nba_client = NBAPlayerStats()
        
        logger.info("Data collection clients initialized successfully")
    
    def collect_data(self, sport: str = "basketball_nba") -> Dict:
        """Collect data from various sources."""
        logger.info(f"Collecting data for {sport}...")
        
        data = {}
        
        try:
            # Collect odds data
            if self.odds_client:
                odds_data = self.odds_client.get_odds(sport)
                if odds_data:
                    data['odds'] = self.odds_client.flatten_odds_data(odds_data)
                    logger.info(f"Collected {len(data['odds'])} odds records")
            
            # Collect NBA player stats
            if self.nba_client and sport == "basketball_nba":
                player_stats = self.nba_client.get_player_season_stats()
                if not player_stats.empty:
                    data['player_stats'] = player_stats
                    logger.info(f"Collected stats for {len(player_stats)} players")
                
                # Get schedule
                schedule = self.nba_client.get_schedule()
                if not schedule.empty:
                    data['schedule'] = schedule
                    logger.info(f"Collected {len(schedule)} scheduled games")
        
        except Exception as e:
            logger.error(f"Error collecting data: {e}")
            return {}
        
        return data
    
    def build_features(self, data: Dict) -> Dict:
        """Build features for machine learning models."""
        logger.info("Building features...")
        
        features = {}
        
        try:
            # Build player features
            if 'player_stats' in data:
                player_features = self.feature_builder.build_player_features(
                    data['player_stats']
                )
                features['player_features'] = player_features
                logger.info(f"Built features for {len(player_features)} player records")
            
            # Build team features
            if 'schedule' in data:
                # This would require more complex team data processing
                logger.info("Team features would be built here with proper team data")
        
        except Exception as e:
            logger.error(f"Error building features: {e}")
            return {}
        
        return features
    
    def train_models(self, features: Dict) -> Dict:
        """Train machine learning models."""
        logger.info("Training models...")
        
        model_results = {}
        
        try:
            # Create baseline models
            models = create_baseline_models()
            
            # For demonstration, we'll use synthetic target data
            # In practice, you'd use actual game outcomes
            if 'player_features' in features:
                X = features['player_features'].select_dtypes(include=[float, int])
                y = X['PTS'] if 'PTS' in X.columns else X.iloc[:, 0]  # Use first column as target
                
                # Remove target from features
                if 'PTS' in X.columns:
                    X = X.drop('PTS', axis=1)
                
                # Train each model
                for name, model in models.items():
                    logger.info(f"Training {name}...")
                    metrics = model.train(X, y)
                    model_results[name] = metrics
                    self.models[name] = model
                
                # Create and train ensemble
                ensemble = ModelEnsemble([models['random_forest'], models['xgboost']])
                ensemble_metrics = ensemble.train(X, y)
                model_results['ensemble'] = ensemble_metrics
                self.models['ensemble'] = ensemble
        
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {}
        
        return model_results
    
    def run_backtest(self, data: Dict, features: Dict) -> Dict:
        """Run backtesting on historical data."""
        logger.info("Running backtest...")
        
        backtest_results = {}
        
        try:
            # Create sample historical data for demonstration
            import pandas as pd
            import numpy as np
            
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
            
            # Test different strategies
            strategies = [
                ConservativeStrategy(),
                AggressiveStrategy()
            ]
            
            for strategy in strategies:
                backtester = Backtester(strategy)
                performance = backtester.run_backtest(historical_data, predictions, odds_data)
                backtest_results[strategy.name] = performance
                logger.info(f"{strategy.name} strategy: ROI = {performance.get('roi_percent', 0):.2f}%")
        
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {}
        
        return backtest_results
    
    def generate_predictions(self, data: Dict, features: Dict) -> Dict:
        """Generate predictions for upcoming games."""
        logger.info("Generating predictions...")
        
        predictions = {}
        
        try:
            if 'player_features' in features and self.models:
                # Use the best model for predictions
                best_model = self.models.get('ensemble', list(self.models.values())[0])
                
                # Prepare features for prediction
                X = features['player_features'].select_dtypes(include=[float, int])
                if 'PTS' in X.columns:
                    X = X.drop('PTS', axis=1)
                
                # Make predictions
                pred_values = best_model.predict(X.head(10))  # Predict for first 10 players
                
                predictions = {
                    'player_predictions': pred_values.tolist(),
                    'model_used': best_model.model_name if hasattr(best_model, 'model_name') else 'ensemble',
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Generated {len(pred_values)} predictions")
        
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return {}
        
        return predictions
    
    def run_full_pipeline(self, sport: str = "basketball_nba") -> Dict:
        """Run the complete pipeline."""
        logger.info("Starting full pipeline execution...")
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'sport': sport,
            'status': 'running'
        }
        
        try:
            # Initialize clients
            self.initialize_data_clients()
            
            # Collect data
            data = self.collect_data(sport)
            pipeline_results['data_collection'] = {
                'status': 'success' if data else 'failed',
                'records_collected': sum(len(v) if hasattr(v, '__len__') else 1 for v in data.values())
            }
            
            if not data:
                pipeline_results['status'] = 'failed'
                return pipeline_results
            
            # Build features
            features = self.build_features(data)
            pipeline_results['feature_engineering'] = {
                'status': 'success' if features else 'failed',
                'features_created': len(features)
            }
            
            if not features:
                pipeline_results['status'] = 'failed'
                return pipeline_results
            
            # Train models
            model_results = self.train_models(features)
            pipeline_results['model_training'] = {
                'status': 'success' if model_results else 'failed',
                'models_trained': len(model_results)
            }
            
            # Run backtest
            backtest_results = self.run_backtest(data, features)
            pipeline_results['backtesting'] = {
                'status': 'success' if backtest_results else 'failed',
                'strategies_tested': len(backtest_results)
            }
            
            # Generate predictions
            predictions = self.generate_predictions(data, features)
            pipeline_results['predictions'] = {
                'status': 'success' if predictions else 'failed',
                'predictions_generated': len(predictions.get('player_predictions', []))
            }
            
            pipeline_results['status'] = 'success'
            pipeline_results['end_time'] = datetime.now().isoformat()
            
            # Store results
            self.results = {
                'data': data,
                'features': features,
                'model_results': model_results,
                'backtest_results': backtest_results,
                'predictions': predictions
            }
            
            logger.info("Pipeline execution completed successfully!")
        
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            pipeline_results['status'] = 'failed'
            pipeline_results['error'] = str(e)
        
        return pipeline_results
    
    def save_results(self, filepath: str = None):
        """Save pipeline results to file."""
        if not filepath:
            filepath = f"results/pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert results to JSON-serializable format
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = value
            elif hasattr(value, 'to_dict'):
                serializable_results[key] = value.to_dict()
            else:
                serializable_results[key] = str(value)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")


def main():
    """Main function to run the pipeline."""
    logger.info("Starting Sports Betting ML Pipeline...")
    
    # Create pipeline
    pipeline = SportsBettingPipeline()
    
    # Run full pipeline
    results = pipeline.run_full_pipeline("basketball_nba")
    
    # Print summary
    print("\n" + "="*50)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*50)
    print(f"Status: {results['status']}")
    print(f"Sport: {results['sport']}")
    print(f"Start Time: {results['start_time']}")
    print(f"End Time: {results.get('end_time', 'N/A')}")
    
    if results['status'] == 'success':
        print("\nResults:")
        for stage, info in results.items():
            if isinstance(info, dict) and 'status' in info:
                print(f"  {stage}: {info['status']}")
        
        # Save results
        pipeline.save_results()
        
        # Print model performance
        if 'model_training' in results and results['model_training']['status'] == 'success':
            print("\nModel Performance:")
            for model_name, metrics in pipeline.results.get('model_results', {}).items():
                if isinstance(metrics, dict) and 'val_r2' in metrics:
                    print(f"  {model_name}: R² = {metrics['val_r2']:.3f}")
        
        # Print backtest results
        if 'backtesting' in results and results['backtesting']['status'] == 'success':
            print("\nBacktest Results:")
            for strategy_name, performance in pipeline.results.get('backtest_results', {}).items():
                if isinstance(performance, dict) and 'roi_percent' in performance:
                    print(f"  {strategy_name}: ROI = {performance['roi_percent']:.2f}%")
    
    else:
        print(f"Pipeline failed: {results.get('error', 'Unknown error')}")
    
    print("="*50)


if __name__ == "__main__":
    main()
