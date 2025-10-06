# 🚀 Quick Start Guide

## Your Next Steps (In Order)

### ✅ Step 1: Collect Training Data
```bash
python3 collect_training_data.py
```
**What it does:**
- Fetches current NBA odds from OddsAPI
- Gets player season stats (569 players)
- Gets team stats (30 teams)
- Collects recent game logs for top 10 players

**Output:** Creates CSV files in `data/` folder

---

### ✅ Step 2: Train Models
```bash
python3 train_models.py
```
**What it does:**
- Builds features from collected data
- Trains RandomForest models for PTS, REB, AST
- Validates model accuracy (R², MAE)
- Saves trained models to `models/` folder

**Expected results:**
- PTS model: R² > 0.6
- REB model: R² > 0.5
- AST model: R² > 0.5

---

### ✅ Step 3: Make Predictions
```bash
python3 make_predictions.py
```
**What it does:**
- Loads trained models
- Makes predictions for top 20 players
- Finds betting opportunities (prediction vs season average)
- Shows OVER/UNDER recommendations

**Output:** `results/predictions_today.csv`

---

## 📊 Understanding the Output

### Predictions Table
```
player              pred_pts  avg_pts  pred_reb  avg_reb
LeBron James        27.5      26.8     8.2       7.9
Stephen Curry       29.1      27.3     5.1       5.3
```

### Betting Opportunities
```
player           stat  prediction  season_avg  difference  recommendation
LeBron James     PTS   27.5        26.8        +0.7        OVER
```

**How to use:**
1. Check if sportsbook line matches `season_avg`
2. If model predicts 10%+ higher → **Bet OVER**
3. If model predicts 10%+ lower → **Bet UNDER**

---

## 🔄 Daily Workflow

Run this every day before games:

```bash
# 1. Get fresh data
python3 collect_training_data.py

# 2. Retrain models (optional, weekly is fine)
python3 train_models.py

# 3. Get predictions
python3 make_predictions.py
```

---

## ⚠️ Current Limitations

1. **Need more historical data** - Currently using recent stats only
2. **No injury data** - Not accounting for player injuries
3. **No opponent adjustments** - Not factoring in defensive matchups
4. **Limited validation** - Need to backtest against actual bet outcomes

---

## 🎯 To Make This Production-Ready

### Short-term (1-2 weeks):
- [ ] Collect 3+ months of historical game logs
- [ ] Add injury report integration
- [ ] Add opponent defensive ratings
- [ ] Implement proper backtesting

### Medium-term (1 month):
- [ ] Add more advanced features (pace, usage rate, rest days)
- [ ] Ensemble multiple model types
- [ ] Track prediction accuracy over time
- [ ] Add bankroll management

### Long-term (2-3 months):
- [ ] Build LSTM models for time-series data
- [ ] Real-time odds scraping
- [ ] Automated bet placement
- [ ] Multi-sport support (NFL, MLB)

---

## 📈 Success Metrics

Track these to measure performance:

- **Model Accuracy**: R² > 0.65 (currently ~0.5-0.7)
- **Prediction Error**: MAE < 3 points (currently ~2-4)
- **Betting ROI**: Target 5%+ (not yet measured)
- **Win Rate**: Target 55%+ (not yet measured)

---

## 🆘 Troubleshooting

**"ModuleNotFoundError"**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**"Data files not found"**
```bash
python3 collect_training_data.py
```

**"Models not trained"**
```bash
python3 train_models.py
```

**"API rate limit exceeded"**
- Wait 60 seconds and try again
- OddsAPI has limited free calls

---

## 💡 Tips

1. **Start small** - Test with top 10-20 players first
2. **Track everything** - Keep a spreadsheet of predictions vs actual
3. **Paper trade first** - Don't bet real money until you've validated accuracy
4. **Stay updated** - Check for injuries and lineup changes
5. **Be patient** - Profitable betting requires hundreds of bets to prove edge

---

## 📞 Next Questions to Answer

- What's the best time to collect data? (2 hours before games)
- Which bookmakers have the softest lines? (need to compare)
- What's the optimal bet size? (Kelly Criterion in backtesting)
- Which stats are most predictable? (PTS > REB > AST currently)

