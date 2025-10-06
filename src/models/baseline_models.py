"""
Baseline machine learning models for sports betting predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SportsBettingModel:
    """Base class for sports betting prediction models."""
    
    def __init__(self, model_name: str, model_type: str = "regression"):
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.training_metrics = {}
        self.is_trained = False
        
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series = None, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess features and target variables."""
        X_processed = X.copy()
        
        # Handle categorical variables
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    X_processed[col] = X_processed[col].astype(str)
                    X_processed[col] = X_processed[col].map(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in self.label_encoders[col].classes_ 
                        else -1
                    )
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.median())
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X_processed)
        else:
            X_scaled = self.scaler.transform(X_processed)
        
        return X_scaled, y.values if y is not None else None
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> Dict[str, float]:
        """Train the model and return performance metrics."""
        logger.info(f"Training {self.model_name}...")
        
        # Preprocess data
        X_processed, y_processed = self.preprocess_data(X, y, fit=True)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y_processed, test_size=validation_split, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        # Calculate metrics
        train_metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred)
        }
        
        val_metrics = {
            'val_mse': mean_squared_error(y_val, y_val_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'val_r2': r2_score(y_val, y_val_pred)
        }
        
        self.training_metrics = {**train_metrics, **val_metrics}
        self.is_trained = True
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        logger.info(f"Training completed. Validation R²: {val_metrics['val_r2']:.3f}")
        return self.training_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_processed, _ = self.preprocess_data(X, fit=False)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions (for classification models)."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.model_name} does not support probability predictions")
        
        X_processed, _ = self.preprocess_data(X, fit=False)
        return self.model.predict_proba(X_processed)
    
    def save_model(self, filepath: str):
        """Save the trained model and preprocessing objects."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and preprocessing objects."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_importance = model_data['feature_importance']
        self.training_metrics = model_data['training_metrics']
        self.model_name = model_data['model_name']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        logger.info(f"Model loaded from {filepath}")


class PlayerPropModel(SportsBettingModel):
    """Model for predicting player prop outcomes."""
    
    def __init__(self, model_name: str = "PlayerPropModel"):
        super().__init__(model_name, "regression")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    
    def predict_prop_probability(self, X: pd.DataFrame, prop_line: float) -> Dict[str, float]:
        """Predict probability of hitting a prop bet."""
        prediction = self.predict(X)[0]
        
        # Calculate probability based on prediction vs line
        # This is a simplified approach - in practice, you'd want more sophisticated probability modeling
        diff = prediction - prop_line
        probability = 1 / (1 + np.exp(-diff * 0.5))  # Sigmoid function
        
        return {
            'predicted_value': prediction,
            'prop_line': prop_line,
            'over_probability': probability,
            'under_probability': 1 - probability,
            'edge': probability - 0.5  # Positive edge if > 0.5
        }


class SpreadModel(SportsBettingModel):
    """Model for predicting game spreads."""
    
    def __init__(self, model_name: str = "SpreadModel"):
        super().__init__(model_name, "regression")
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    
    def predict_spread(self, X: pd.DataFrame) -> Dict[str, float]:
        """Predict the actual spread of a game."""
        prediction = self.predict(X)[0]
        
        return {
            'predicted_spread': prediction,
            'confidence': min(abs(prediction) / 10, 1.0)  # Confidence based on spread magnitude
        }


class TotalModel(SportsBettingModel):
    """Model for predicting game totals (over/under)."""
    
    def __init__(self, model_name: str = "TotalModel"):
        super().__init__(model_name, "regression")
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    
    def predict_total(self, X: pd.DataFrame) -> Dict[str, float]:
        """Predict the total points scored in a game."""
        prediction = self.predict(X)[0]
        
        return {
            'predicted_total': prediction,
            'confidence': min(abs(prediction - 220) / 50, 1.0)  # Confidence based on deviation from average
        }


class ModelEnsemble:
    """Ensemble of multiple models for improved predictions."""
    
    def __init__(self, models: List[SportsBettingModel], weights: List[float] = None):
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        self.is_trained = False
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> Dict[str, float]:
        """Train all models in the ensemble."""
        logger.info("Training ensemble models...")
        
        ensemble_metrics = {}
        
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}: {model.model_name}")
            metrics = model.train(X, y, validation_split)
            ensemble_metrics[f"{model.model_name}_val_r2"] = metrics['val_r2']
        
        self.is_trained = True
        return ensemble_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_pred
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate mean and standard deviation
        mean_pred = np.average(predictions, axis=0, weights=self.weights)
        std_pred = np.sqrt(np.average((predictions - mean_pred) ** 2, axis=0, weights=self.weights))
        
        return {
            'prediction': mean_pred,
            'uncertainty': std_pred,
            'individual_predictions': predictions
        }


def create_baseline_models() -> Dict[str, SportsBettingModel]:
    """Create a set of baseline models for different prediction tasks."""
    models = {
        'player_props': PlayerPropModel(),
        'spreads': SpreadModel(),
        'totals': TotalModel(),
        'random_forest': SportsBettingModel("RandomForest", "regression"),
        'xgboost': SportsBettingModel("XGBoost", "regression"),
        'linear_regression': SportsBettingModel("LinearRegression", "regression")
    }
    
    # Set specific models
    models['random_forest'].model = RandomForestRegressor(n_estimators=100, random_state=42)
    models['xgboost'].model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    models['linear_regression'].model = LinearRegression()
    
    return models


def main():
    """Example usage of the baseline models."""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'categorical': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    y = pd.Series(
        2 * X['feature1'] + 1.5 * X['feature2'] - X['feature3'] + 
        np.random.normal(0, 0.5, n_samples)
    )
    
    # Create and train models
    models = create_baseline_models()
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        metrics = model.train(X, y)
        print(f"Validation R²: {metrics['val_r2']:.3f}")
        
        # Make predictions
        predictions = model.predict(X.head(5))
        print(f"Sample predictions: {predictions}")
    
    # Test ensemble
    ensemble = ModelEnsemble([models['random_forest'], models['xgboost']])
    ensemble.train(X, y)
    ensemble_pred = ensemble.predict(X.head(5))
    print(f"\nEnsemble predictions: {ensemble_pred}")


if __name__ == "__main__":
    main()
