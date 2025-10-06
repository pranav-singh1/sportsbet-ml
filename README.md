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
