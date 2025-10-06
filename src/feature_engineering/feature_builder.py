"""
Feature engineering module for sports betting ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SportsFeatureBuilder:
    """Build features for sports betting models."""
    
    def __init__(self):
        self.feature_cache = {}
    
    def build_player_features(
        self, 
        player_stats: pd.DataFrame, 
        opponent_stats: pd.DataFrame = None,
        market_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Build comprehensive features for player prop predictions."""
        
        features = player_stats.copy()
        
        # Rolling averages (last 5, 10, 15 games)
        features = self._add_rolling_averages(features, ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV'])
        
        # Performance trends
        features = self._add_performance_trends(features, ['PTS', 'REB', 'AST'])
        
        # Rest days and back-to-back games
        features = self._add_rest_features(features)
        
        # Home/away splits
        features = self._add_home_away_features(features)
        
        # Opponent adjustments
        if opponent_stats is not None:
            features = self._add_opponent_adjustments(features, opponent_stats)
        
        # Market features
        if market_data is not None:
            features = self._add_market_features(features, market_data)
        
        # Advanced metrics
        features = self._add_advanced_metrics(features)
        
        return features
    
    def build_team_features(
        self, 
        team_stats: pd.DataFrame,
        opponent_stats: pd.DataFrame = None,
        market_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Build features for team-based predictions (spreads, totals)."""
        
        features = team_stats.copy()
        
        # Team performance metrics
        features = self._add_team_performance_features(features)
        
        # Pace and efficiency
        features = self._add_pace_efficiency_features(features)
        
        # Defensive metrics
        features = self._add_defensive_features(features)
        
        # Opponent matchup features
        if opponent_stats is not None:
            features = self._add_team_matchup_features(features, opponent_stats)
        
        # Market features
        if market_data is not None:
            features = self._add_team_market_features(features, market_data)
        
        return features
    
    def _add_rolling_averages(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Add rolling averages for specified columns."""
        for col in columns:
            if col in df.columns:
                df[f'{col}_avg_5'] = df[col].rolling(window=5, min_periods=1).mean()
                df[f'{col}_avg_10'] = df[col].rolling(window=10, min_periods=1).mean()
                df[f'{col}_avg_15'] = df[col].rolling(window=15, min_periods=1).mean()
                
                # Rolling standard deviation for consistency
                df[f'{col}_std_5'] = df[col].rolling(window=5, min_periods=1).std()
                df[f'{col}_std_10'] = df[col].rolling(window=10, min_periods=1).std()
        return df
    
    def _add_performance_trends(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Add performance trend indicators."""
        for col in columns:
            if col in df.columns:
                # Recent vs overall performance
                df[f'{col}_trend_3v10'] = (
                    df[col].rolling(window=3, min_periods=1).mean() / 
                    df[col].rolling(window=10, min_periods=1).mean()
                )
                
                # Momentum indicator
                df[f'{col}_momentum'] = df[col].diff(3).rolling(window=5, min_periods=1).mean()
                
                # Streak indicators
                df[f'{col}_above_avg'] = (df[col] > df[col].rolling(window=10, min_periods=1).mean()).astype(int)
        return df
    
    def _add_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rest day and back-to-back features."""
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values('GAME_DATE')
            
            # Rest days
            df['rest_days'] = df['GAME_DATE'].diff().dt.days.fillna(0)
            
            # Back-to-back indicator
            df['is_back_to_back'] = (df['rest_days'] == 1).astype(int)
            
            # Rest day categories
            df['rest_category'] = pd.cut(
                df['rest_days'], 
                bins=[-1, 0, 1, 2, 7, float('inf')], 
                labels=['same_day', 'back_to_back', '1_day', '2-7_days', '7+_days']
            )
        return df
    
    def _add_home_away_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add home/away performance features."""
        if 'MATCHUP' in df.columns:
            df['is_home'] = df['MATCHUP'].str.contains('vs.').astype(int)
            df['is_away'] = df['MATCHUP'].str.contains('@').astype(int)
            
            # Home/away splits for key stats
            for col in ['PTS', 'REB', 'AST']:
                if col in df.columns:
                    home_avg = df[df['is_home'] == 1][col].mean()
                    away_avg = df[df['is_away'] == 1][col].mean()
                    df[f'{col}_home_advantage'] = home_avg - away_avg
        return df
    
    def _add_opponent_adjustments(self, df: pd.DataFrame, opponent_stats: pd.DataFrame) -> pd.DataFrame:
        """Add opponent-specific adjustments."""
        # This would require more complex logic to match opponents
        # For now, add placeholder features
        df['opponent_def_rating'] = np.random.normal(100, 10, len(df))  # Placeholder
        df['opponent_pace'] = np.random.normal(100, 5, len(df))  # Placeholder
        return df
    
    def _add_market_features(self, df: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Add market-based features."""
        # Line movement features
        df['line_movement'] = np.random.normal(0, 2, len(df))  # Placeholder
        df['public_betting_pct'] = np.random.uniform(0.3, 0.7, len(df))  # Placeholder
        df['sharp_money_indicator'] = (df['public_betting_pct'] < 0.4).astype(int)
        return df
    
    def _add_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced basketball metrics."""
        if all(col in df.columns for col in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGM', 'FGA']):
            # Player Efficiency Rating (simplified)
            df['per_simplified'] = (
                df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK'] - df['TOV']
            )
            
            # True Shooting Percentage
            if 'FTM' in df.columns and 'FTA' in df.columns:
                df['ts_pct'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
            
            # Usage rate approximation
            if 'MIN' in df.columns:
                df['usage_rate_approx'] = (df['FGA'] + df['TOV']) / df['MIN'] * 100
        
        return df
    
    def _add_team_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team performance features."""
        # Win percentage
        if 'W' in df.columns and 'L' in df.columns:
            df['win_pct'] = df['W'] / (df['W'] + df['L'])
        
        # Point differential
        if 'PTS' in df.columns and 'OPP_PTS' in df.columns:
            df['point_diff'] = df['PTS'] - df['OPP_PTS']
            df['point_diff_per_game'] = df['point_diff'] / df['GP']
        
        return df
    
    def _add_pace_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pace and efficiency features."""
        # Pace (possessions per game)
        if all(col in df.columns for col in ['FGA', 'FTA', 'TOV', 'OREB', 'OPP_DREB']):
            df['pace'] = (
                df['FGA'] + 0.44 * df['FTA'] + df['TOV'] - df['OREB'] + df['OPP_DREB']
            )
        
        # Offensive and defensive efficiency
        if 'PTS' in df.columns and 'pace' in df.columns:
            df['off_eff'] = df['PTS'] / df['pace'] * 100
        if 'OPP_PTS' in df.columns and 'pace' in df.columns:
            df['def_eff'] = df['OPP_PTS'] / df['pace'] * 100
        
        return df
    
    def _add_defensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add defensive-specific features."""
        if all(col in df.columns for col in ['STL', 'BLK', 'OPP_PTS']):
            df['def_rating'] = df['OPP_PTS'] / df['GP']  # Points allowed per game
            df['steal_rate'] = df['STL'] / df['GP']
            df['block_rate'] = df['BLK'] / df['GP']
        
        return df
    
    def _add_team_matchup_features(self, df: pd.DataFrame, opponent_stats: pd.DataFrame) -> pd.DataFrame:
        """Add team matchup features."""
        # Placeholder for complex opponent matching logic
        df['opponent_strength'] = np.random.normal(0.5, 0.1, len(df))
        df['matchup_advantage'] = np.random.normal(0, 0.05, len(df))
        return df
    
    def _add_team_market_features(self, df: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Add team market features."""
        df['spread_movement'] = np.random.normal(0, 1, len(df))
        df['total_movement'] = np.random.normal(0, 1, len(df))
        df['public_side'] = np.random.uniform(0.3, 0.7, len(df))
        return df
    
    def create_prop_prediction_features(
        self, 
        player_id: str, 
        stat_type: str, 
        historical_data: pd.DataFrame
    ) -> Dict:
        """Create features specifically for prop bet predictions."""
        
        # Filter data for specific player and stat
        player_data = historical_data[historical_data['PLAYER_ID'] == player_id].copy()
        
        if player_data.empty:
            return {}
        
        # Sort by date
        player_data = player_data.sort_values('GAME_DATE')
        
        # Calculate features
        features = {
            'player_id': player_id,
            'stat_type': stat_type,
            'recent_avg_5': player_data[stat_type].tail(5).mean(),
            'recent_avg_10': player_data[stat_type].tail(10).mean(),
            'season_avg': player_data[stat_type].mean(),
            'consistency': 1 - (player_data[stat_type].std() / player_data[stat_type].mean()),
            'trend_slope': self._calculate_trend_slope(player_data[stat_type].tail(10)),
            'home_avg': player_data[player_data['is_home'] == 1][stat_type].mean(),
            'away_avg': player_data[player_data['is_away'] == 1][stat_type].mean(),
            'vs_opponent_avg': player_data[stat_type].mean(),  # Would need opponent matching
            'rest_days_avg': player_data['rest_days'].mean() if 'rest_days' in player_data.columns else 0,
            'back_to_back_pct': (player_data['is_back_to_back'] == 1).mean() if 'is_back_to_back' in player_data.columns else 0
        }
        
        return features
    
    def _calculate_trend_slope(self, series: pd.Series) -> float:
        """Calculate the slope of a trend line."""
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        y = series.values
        
        # Simple linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return round(slope, 3)


def main():
    """Example usage of SportsFeatureBuilder."""
    # Create sample data
    sample_data = pd.DataFrame({
        'PLAYER_ID': ['player1'] * 20,
        'GAME_DATE': pd.date_range('2024-01-01', periods=20, freq='D'),
        'PTS': np.random.normal(20, 5, 20),
        'REB': np.random.normal(8, 3, 20),
        'AST': np.random.normal(6, 2, 20),
        'STL': np.random.normal(1.5, 0.5, 20),
        'BLK': np.random.normal(1, 0.5, 20),
        'TOV': np.random.normal(3, 1, 20),
        'MATCHUP': ['vs. Team A', '@ Team B'] * 10
    })
    
    # Build features
    feature_builder = SportsFeatureBuilder()
    features = feature_builder.build_player_features(sample_data)
    
    print("Sample features created:")
    print(features.columns.tolist())
    print(features[['PTS', 'PTS_avg_5', 'PTS_trend_3v10', 'rest_days']].head())


if __name__ == "__main__":
    main()
