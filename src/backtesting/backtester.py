"""
Backtesting framework for sports betting strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BettingStrategy:
    """Base class for betting strategies."""
    
    def __init__(self, name: str, initial_bankroll: float = 10000):
        self.name = name
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.bets = []
        self.performance_metrics = {}
    
    def should_bet(self, prediction: Dict, odds: Dict, bankroll: float) -> Tuple[bool, float]:
        """Determine if a bet should be placed and how much to bet."""
        raise NotImplementedError("Subclasses must implement should_bet method")
    
    def calculate_bet_size(self, edge: float, odds: float, bankroll: float) -> float:
        """Calculate optimal bet size using Kelly Criterion or similar."""
        # Kelly Criterion: f = (bp - q) / b
        # where b = odds - 1, p = probability of winning, q = 1 - p
        if odds <= 1:
            return 0
        
        b = odds - 1
        p = 1 / odds  # Implied probability from odds
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Cap at 5% of bankroll for safety
        max_bet = bankroll * 0.05
        kelly_bet = bankroll * kelly_fraction
        
        return min(kelly_bet, max_bet) if kelly_fraction > 0 else 0


class ConservativeStrategy(BettingStrategy):
    """Conservative betting strategy with low risk tolerance."""
    
    def __init__(self, initial_bankroll: float = 10000, min_edge: float = 0.05):
        super().__init__("Conservative", initial_bankroll)
        self.min_edge = min_edge
        self.max_bet_pct = 0.02  # Max 2% of bankroll per bet
    
    def should_bet(self, prediction: Dict, odds: Dict, bankroll: float) -> Tuple[bool, float]:
        """Conservative betting logic."""
        edge = prediction.get('edge', 0)
        
        if edge < self.min_edge:
            return False, 0
        
        # Use conservative bet sizing
        bet_size = bankroll * self.max_bet_pct
        return True, bet_size


class AggressiveStrategy(BettingStrategy):
    """Aggressive betting strategy with higher risk tolerance."""
    
    def __init__(self, initial_bankroll: float = 10000, min_edge: float = 0.02):
        super().__init__("Aggressive", initial_bankroll)
        self.min_edge = min_edge
    
    def should_bet(self, prediction: Dict, odds: Dict, bankroll: float) -> Tuple[bool, float]:
        """Aggressive betting logic using Kelly Criterion."""
        edge = prediction.get('edge', 0)
        
        if edge < self.min_edge:
            return False, 0
        
        # Use Kelly Criterion for bet sizing
        odds_decimal = odds.get('decimal', 2.0)
        bet_size = self.calculate_bet_size(edge, odds_decimal, bankroll)
        
        return True, bet_size


class Backtester:
    """Main backtesting engine for sports betting strategies."""
    
    def __init__(self, strategy: BettingStrategy):
        self.strategy = strategy
        self.results = []
        self.performance_history = []
    
    def run_backtest(
        self, 
        historical_data: pd.DataFrame,
        predictions: pd.DataFrame,
        odds_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run backtest on historical data."""
        logger.info(f"Running backtest for {self.strategy.name} strategy...")
        
        # Merge data
        merged_data = self._merge_data(historical_data, predictions, odds_data)
        
        # Reset strategy state
        self.strategy.current_bankroll = self.strategy.initial_bankroll
        self.strategy.bets = []
        
        # Process each game
        for _, row in merged_data.iterrows():
            self._process_bet(row)
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics()
        
        logger.info(f"Backtest completed. Final bankroll: ${self.strategy.current_bankroll:.2f}")
        return performance
    
    def _merge_data(
        self, 
        historical_data: pd.DataFrame, 
        predictions: pd.DataFrame, 
        odds_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge historical data, predictions, and odds."""
        # This is a simplified merge - in practice, you'd need more sophisticated matching
        merged = historical_data.copy()
        
        # Add prediction columns
        for col in predictions.columns:
            if col not in merged.columns:
                merged[col] = predictions[col]
        
        # Add odds columns
        for col in odds_data.columns:
            if col not in merged.columns:
                merged[col] = odds_data[col]
        
        return merged
    
    def _process_bet(self, row: pd.Series):
        """Process a single bet."""
        # Extract prediction and odds
        prediction = {
            'predicted_value': row.get('predicted_value', 0),
            'probability': row.get('probability', 0.5),
            'edge': row.get('edge', 0)
        }
        
        odds = {
            'decimal': row.get('odds_decimal', 2.0),
            'american': row.get('odds_american', 100)
        }
        
        # Determine if bet should be placed
        should_bet, bet_size = self.strategy.should_bet(
            prediction, odds, self.strategy.current_bankroll
        )
        
        if should_bet and bet_size > 0:
            # Place bet
            bet = {
                'game_id': row.get('game_id', 'unknown'),
                'date': row.get('date', datetime.now()),
                'bet_type': row.get('bet_type', 'unknown'),
                'prediction': prediction,
                'odds': odds,
                'bet_size': bet_size,
                'bankroll_before': self.strategy.current_bankroll
            }
            
            # Simulate bet outcome (in practice, you'd use actual game results)
            outcome = self._simulate_bet_outcome(prediction, row)
            bet['outcome'] = outcome
            bet['profit'] = self._calculate_profit(bet_size, odds['decimal'], outcome)
            
            # Update bankroll
            self.strategy.current_bankroll += bet['profit']
            bet['bankroll_after'] = self.strategy.current_bankroll
            
            self.strategy.bets.append(bet)
    
    def _simulate_bet_outcome(self, prediction: Dict, row: pd.Series) -> bool:
        """Simulate the outcome of a bet (win/loss)."""
        # In practice, this would use actual game results
        # For now, use the predicted probability to simulate outcomes
        probability = prediction.get('probability', 0.5)
        return np.random.random() < probability
    
    def _calculate_profit(self, bet_size: float, odds: float, won: bool) -> float:
        """Calculate profit from a bet."""
        if won:
            return bet_size * (odds - 1)
        else:
            return -bet_size
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not self.strategy.bets:
            return {'error': 'No bets placed'}
        
        bets_df = pd.DataFrame(self.strategy.bets)
        
        # Basic metrics
        total_bets = len(bets_df)
        winning_bets = len(bets_df[bets_df['outcome'] == True])
        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        
        total_profit = sum(bet['profit'] for bet in self.strategy.bets)
        roi = (total_profit / self.strategy.initial_bankroll) * 100
        
        # Risk metrics
        profits = [bet['profit'] for bet in self.strategy.bets]
        avg_profit = np.mean(profits)
        profit_std = np.std(profits)
        sharpe_ratio = avg_profit / profit_std if profit_std > 0 else 0
        
        # Drawdown analysis
        bankroll_history = [bet['bankroll_after'] for bet in self.strategy.bets]
        peak = self.strategy.initial_bankroll
        max_drawdown = 0
        
        for bankroll in bankroll_history:
            if bankroll > peak:
                peak = bankroll
            drawdown = (peak - bankroll) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Kelly Criterion analysis
        avg_odds = np.mean([bet['odds']['decimal'] for bet in self.strategy.bets])
        kelly_optimal = self._calculate_kelly_optimal(win_rate, avg_odds)
        
        metrics = {
            'strategy_name': self.strategy.name,
            'initial_bankroll': self.strategy.initial_bankroll,
            'final_bankroll': self.strategy.current_bankroll,
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'roi_percent': roi,
            'avg_profit_per_bet': avg_profit,
            'profit_std': profit_std,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'kelly_optimal': kelly_optimal,
            'avg_odds': avg_odds
        }
        
        return metrics
    
    def _calculate_kelly_optimal(self, win_rate: float, avg_odds: float) -> float:
        """Calculate optimal Kelly fraction."""
        if avg_odds <= 1:
            return 0
        
        b = avg_odds - 1
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        return max(0, kelly_fraction)
    
    def plot_performance(self, save_path: Optional[str] = None):
        """Plot performance charts."""
        if not self.strategy.bets:
            logger.warning("No bets to plot")
            return
        
        bets_df = pd.DataFrame(self.strategy.bets)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.strategy.name} Strategy Performance', fontsize=16)
        
        # Bankroll over time
        axes[0, 0].plot(bets_df['bankroll_after'])
        axes[0, 0].set_title('Bankroll Over Time')
        axes[0, 0].set_xlabel('Bet Number')
        axes[0, 0].set_ylabel('Bankroll ($)')
        axes[0, 0].grid(True)
        
        # Profit distribution
        axes[0, 1].hist(bets_df['profit'], bins=20, alpha=0.7)
        axes[0, 1].set_title('Profit Distribution')
        axes[0, 1].set_xlabel('Profit ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True)
        
        # Win rate over time (rolling)
        rolling_wins = bets_df['outcome'].rolling(window=20).mean()
        axes[1, 0].plot(rolling_wins)
        axes[1, 0].set_title('Rolling Win Rate (20 bets)')
        axes[1, 0].set_xlabel('Bet Number')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].grid(True)
        
        # Bet size over time
        axes[1, 1].plot(bets_df['bet_size'])
        axes[1, 1].set_title('Bet Size Over Time')
        axes[1, 1].set_xlabel('Bet Number')
        axes[1, 1].set_ylabel('Bet Size ($)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """Generate a comprehensive performance report."""
        metrics = self._calculate_performance_metrics()
        
        if 'error' in metrics:
            return f"Error: {metrics['error']}"
        
        report = f"""
=== {metrics['strategy_name']} Strategy Performance Report ===

Bankroll:
  Initial: ${metrics['initial_bankroll']:,.2f}
  Final: ${metrics['final_bankroll']:,.2f}
  Total Profit: ${metrics['total_profit']:,.2f}
  ROI: {metrics['roi_percent']:.2f}%

Betting Statistics:
  Total Bets: {metrics['total_bets']}
  Winning Bets: {metrics['winning_bets']}
  Win Rate: {metrics['win_rate']:.2%}
  Average Profit per Bet: ${metrics['avg_profit_per_bet']:.2f}

Risk Metrics:
  Profit Standard Deviation: ${metrics['profit_std']:.2f}
  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
  Maximum Drawdown: {metrics['max_drawdown']:.2%}
  Kelly Optimal Fraction: {metrics['kelly_optimal']:.3f}

Market Analysis:
  Average Odds: {metrics['avg_odds']:.2f}
"""
        
        return report


def compare_strategies(
    strategies: List[BettingStrategy],
    historical_data: pd.DataFrame,
    predictions: pd.DataFrame,
    odds_data: pd.DataFrame
) -> pd.DataFrame:
    """Compare multiple betting strategies."""
    results = []
    
    for strategy in strategies:
        backtester = Backtester(strategy)
        performance = backtester.run_backtest(historical_data, predictions, odds_data)
        results.append(performance)
    
    return pd.DataFrame(results)


def main():
    """Example usage of the backtesting framework."""
    # Create sample data
    np.random.seed(42)
    n_games = 1000
    
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
    
    # Test strategies
    conservative = ConservativeStrategy()
    aggressive = AggressiveStrategy()
    
    strategies = [conservative, aggressive]
    comparison = compare_strategies(strategies, historical_data, predictions, odds_data)
    
    print("Strategy Comparison:")
    print(comparison[['strategy_name', 'roi_percent', 'win_rate', 'sharpe_ratio', 'max_drawdown']])
    
    # Generate detailed report for best strategy
    best_strategy = comparison.loc[comparison['roi_percent'].idxmax()]
    backtester = Backtester(conservative)  # Use conservative for demo
    backtester.run_backtest(historical_data, predictions, odds_data)
    report = backtester.generate_report()
    print(report)


if __name__ == "__main__":
    main()
