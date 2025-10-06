"""
Configuration settings for the sports betting ML system.
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
if not ODDS_API_KEY:
    raise ValueError("ODDS_API_KEY not found! Create a .env file with your API key.")
NBA_API_BASE_URL = "https://stats.nba.com/stats"

# Data Collection Settings
DATA_COLLECTION = {
    "sports": ["basketball_nba", "americanfootball_nfl", "baseball_mlb"],
    "regions": ["us"],
    "markets": ["h2h", "spreads", "totals"],
    "odds_format": "american",
    "update_frequency": 300,  # seconds
    "historical_days": 30
}

# Feature Engineering Settings
FEATURE_ENGINEERING = {
    "rolling_windows": [5, 10, 15, 20],
    "trend_periods": [3, 5, 10],
    "min_games_for_features": 5,
    "categorical_encoding": "label",
    "missing_value_strategy": "median"
}

# Model Configuration
MODELS = {
    "baseline": {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        }
    },
    "advanced": {
        "lstm": {
            "sequence_length": 10,
            "hidden_units": 64,
            "dropout": 0.2,
            "epochs": 100,
            "batch_size": 32
        },
        "transformer": {
            "d_model": 128,
            "n_heads": 8,
            "n_layers": 4,
            "dropout": 0.1,
            "epochs": 50,
            "batch_size": 16
        }
    }
}

# Backtesting Configuration
BACKTESTING = {
    "initial_bankroll": 10000,
    "strategies": {
        "conservative": {
            "min_edge": 0.05,
            "max_bet_pct": 0.02,
            "kelly_fraction": 0.25
        },
        "aggressive": {
            "min_edge": 0.02,
            "max_bet_pct": 0.05,
            "kelly_fraction": 0.5
        },
        "kelly_optimal": {
            "min_edge": 0.01,
            "max_bet_pct": 0.1,
            "kelly_fraction": 1.0
        }
    },
    "performance_metrics": [
        "roi", "sharpe_ratio", "max_drawdown", "win_rate", "profit_factor"
    ]
}

# Deployment Configuration
DEPLOYMENT = {
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 4,
        "reload": True
    },
    "database": {
        "url": os.getenv("DATABASE_URL", "sqlite:///sportsbet.db"),
        "pool_size": 10,
        "max_overflow": 20
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "logs/sportsbet.log"
    }
}

# Notification Settings
NOTIFICATIONS = {
    "telegram": {
        "bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
        "chat_id": os.getenv("TELEGRAM_CHAT_ID"),
        "enabled": False
    },
    "discord": {
        "webhook_url": os.getenv("DISCORD_WEBHOOK_URL"),
        "enabled": False
    },
    "email": {
        "smtp_server": os.getenv("SMTP_SERVER"),
        "smtp_port": int(os.getenv("SMTP_PORT", 587)),
        "username": os.getenv("EMAIL_USERNAME"),
        "password": os.getenv("EMAIL_PASSWORD"),
        "enabled": False
    }
}

# File Paths
PATHS = {
    "data": "data/",
    "models": "models/",
    "logs": "logs/",
    "notebooks": "notebooks/",
    "reports": "reports/"
}

# Sports-specific Settings
SPORTS_CONFIG = {
    "basketball_nba": {
        "season_months": [10, 11, 12, 1, 2, 3, 4, 5, 6],
        "playoff_months": [4, 5, 6],
        "key_stats": ["PTS", "REB", "AST", "STL", "BLK", "TOV"],
        "prop_markets": ["points", "rebounds", "assists", "steals", "blocks", "turnovers"],
        "game_duration": 48,  # minutes
        "avg_possessions": 100
    },
    "americanfootball_nfl": {
        "season_months": [9, 10, 11, 12, 1],
        "playoff_months": [1],
        "key_stats": ["PASS_YDS", "RUSH_YDS", "REC_YDS", "TD", "INT"],
        "prop_markets": ["passing_yards", "rushing_yards", "receiving_yards", "touchdowns"],
        "game_duration": 60,  # minutes
        "avg_possessions": 12
    },
    "baseball_mlb": {
        "season_months": [3, 4, 5, 6, 7, 8, 9, 10],
        "playoff_months": [10],
        "key_stats": ["H", "HR", "RBI", "R", "SB"],
        "prop_markets": ["hits", "home_runs", "rbis", "runs", "stolen_bases"],
        "game_duration": 180,  # minutes
        "avg_innings": 9
    }
}

# Model Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    "min_accuracy": 0.55,
    "min_precision": 0.60,
    "min_recall": 0.50,
    "min_f1_score": 0.55,
    "min_roi": 0.05,  # 5% ROI
    "max_drawdown": 0.20,  # 20% max drawdown
    "min_sharpe_ratio": 1.0
}

# Risk Management
RISK_MANAGEMENT = {
    "max_daily_bets": 10,
    "max_bet_percentage": 0.05,  # 5% of bankroll
    "stop_loss_percentage": 0.15,  # 15% stop loss
    "take_profit_percentage": 0.30,  # 30% take profit
    "correlation_limit": 0.7,  # Max correlation between bets
    "diversification_required": True
}

# Data Quality Checks
DATA_QUALITY = {
    "min_games_per_player": 10,
    "max_missing_percentage": 0.1,  # 10% max missing data
    "outlier_threshold": 3,  # 3 standard deviations
    "consistency_checks": True,
    "duplicate_detection": True
}

# Monitoring and Alerting
MONITORING = {
    "performance_check_interval": 3600,  # 1 hour
    "model_drift_threshold": 0.05,
    "data_freshness_threshold": 1800,  # 30 minutes
    "alert_on_anomalies": True,
    "alert_on_failures": True
}

def get_config(section: str) -> Dict:
    """Get configuration for a specific section."""
    config_map = {
        "data_collection": DATA_COLLECTION,
        "feature_engineering": FEATURE_ENGINEERING,
        "models": MODELS,
        "backtesting": BACKTESTING,
        "deployment": DEPLOYMENT,
        "notifications": NOTIFICATIONS,
        "paths": PATHS,
        "sports": SPORTS_CONFIG,
        "performance": PERFORMANCE_THRESHOLDS,
        "risk": RISK_MANAGEMENT,
        "data_quality": DATA_QUALITY,
        "monitoring": MONITORING
    }
    
    return config_map.get(section, {})

def validate_config() -> List[str]:
    """Validate configuration settings and return any issues."""
    issues = []
    
    # Check required API keys
    if not ODDS_API_KEY or ODDS_API_KEY == "your_api_key_here":
        issues.append("ODDS_API_KEY not set or using placeholder value")
    
    # Check file paths exist
    for path_name, path_value in PATHS.items():
        if not os.path.exists(path_value):
            issues.append(f"Path {path_name}: {path_value} does not exist")
    
    # Validate model parameters
    for model_type, models in MODELS.items():
        for model_name, params in models.items():
            if "random_state" in params and params["random_state"] is None:
                issues.append(f"Model {model_name} missing random_state")
    
    # Validate backtesting parameters
    if BACKTESTING["initial_bankroll"] <= 0:
        issues.append("Initial bankroll must be positive")
    
    for strategy_name, strategy_params in BACKTESTING["strategies"].items():
        if strategy_params["max_bet_pct"] > 0.1:
            issues.append(f"Strategy {strategy_name} has high max bet percentage")
    
    return issues

if __name__ == "__main__":
    # Validate configuration
    issues = validate_config()
    if issues:
        print("Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration is valid!")
    
    # Print sample configuration
    print("\nSample Configuration:")
    print(f"Odds API Key: {ODDS_API_KEY[:10]}..." if ODDS_API_KEY else "Not set")
    print(f"Initial Bankroll: ${BACKTESTING['initial_bankroll']:,}")
    print(f"Available Sports: {', '.join(DATA_COLLECTION['sports'])}")
