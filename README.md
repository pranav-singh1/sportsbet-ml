# 🧠 Sports Betting ML System

This project builds a machine learning pipeline for predicting sports outcomes and player prop probabilities.
The goal is to uncover exploitable inefficiencies in sportsbook markets and automate profitable betting strategies.

## 🎯 Objective

Leverage data-driven models to predict outcomes more accurately than sportsbooks, focusing on:

- Player props (points, rebounds, assists, passing yards, etc.)
- Game totals and spreads
- Real-time odds movement and line inefficiencies

## ⚙️ System Overview

### Data Collection
- Pull odds and player stats via APIs (OddsAPI, nba_api, etc.)
- Scrape injury reports, team news, and public betting percentages
- Build structured historical datasets for training

### Feature Engineering
- Rolling averages, opponent adjustments, rest days, team pace
- Player usage rate, minutes projections, weather (for NFL/MLB)
- Market data features (line movement, implied probabilities)

### Modeling
- Baseline: RandomForest / XGBoost regression
- Advanced: LSTM / transformer sequence model for time-series data
- Output: probability distributions for prop outcomes and win margins

### Backtesting
- Simulate past bets vs sportsbook odds
- Track profit curves, Sharpe ratios, and edge consistency

### Deployment
- Automated odds fetch → model prediction → bet signal generation
- (Optional) Integration with Telegram or Discord for auto-alerts
- Potential bankroll-management module for live play

## 🧩 Tech Stack

- **Python** / Jupyter / VS Code
- **Libraries**: pandas, NumPy, scikit-learn, XGBoost, PyTorch
- **APIs**: OddsAPI, nba_api, ESPN, The OddsDB
- **Storage**: Supabase / PostgreSQL
- **Deployment**: FastAPI + Render / Railway

## 💡 Long-Term Goal

Create a fully automated sports-betting AI agent that:

- Continuously learns from new games
- Self-optimizes feature weights based on profit outcomes
- Detects new edges faster than human analysts

## 📈 Vision

A data-driven betting intelligence engine capable of generating consistent, market-beating ROI across sports and seasons.

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pranav-singh1/sportsbet-ml.git
   cd sportsbet-ml
   ```

2. **Set up the environment:**
   ```bash
   # Create virtual environment (recommended)
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   python3 setup.py
   ```

3. **Configure API keys:**
   ```bash
   # Edit config/config.py and add your API keys
   # - ODDS_API_KEY: Get from https://the-odds-api.com/
   # - Other API keys as needed
   ```

### Usage

1. **Run the example script:**
   ```bash
   python3 example_usage.py
   ```

2. **Run the full pipeline:**
   ```bash
   python3 src/main.py
   ```

3. **Explore Jupyter notebooks:**
   ```bash
   jupyter lab notebooks/
   ```

## 📁 Project Structure

```
sportsbet-ml/
├── src/                    # Source code
│   ├── data_collection/    # API clients and data fetching
│   ├── feature_engineering/ # Feature building and preprocessing
│   ├── models/            # ML models and training
│   ├── backtesting/       # Strategy testing and validation
│   └── main.py           # Main pipeline
├── data/                  # Data storage
├── models/               # Trained model storage
├── notebooks/            # Jupyter notebooks
├── config/               # Configuration files
├── results/              # Output results
├── logs/                 # Log files
├── example_usage.py      # Example usage script
├── setup.py             # Setup script
└── requirements.txt     # Dependencies
```

## 🔧 Configuration

Edit `config/config.py` to customize:

- **API Keys**: Add your OddsAPI and other API keys
- **Model Parameters**: Adjust ML model settings
- **Backtesting**: Configure betting strategies
- **Risk Management**: Set betting limits and rules

## 📊 Features Implemented

✅ **Data Collection**
- OddsAPI integration for real-time odds
- NBA API for player and team statistics
- Historical data collection and storage

✅ **Feature Engineering**
- Rolling averages and trends
- Rest days and back-to-back analysis
- Home/away performance splits
- Market-based features

✅ **Machine Learning Models**
- Random Forest and XGBoost
- Ensemble methods
- Model evaluation and validation

✅ **Backtesting Framework**
- Multiple betting strategies
- Performance metrics (ROI, Sharpe ratio, drawdown)
- Risk management tools

## 🎯 Next Steps

- [ ] Advanced LSTM/Transformer models
- [ ] Real-time prediction API
- [ ] Telegram/Discord notifications
- [ ] Live betting integration
- [ ] Multi-sport support (NFL, MLB, etc.)

## ⚠️ Disclaimer

This system is for educational and research purposes only. Sports betting involves risk, and past performance does not guarantee future results. Please bet responsibly and within your means.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
