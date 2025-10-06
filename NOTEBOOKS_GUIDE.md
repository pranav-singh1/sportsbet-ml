# 📓 Jupyter Notebooks Guide

## Why Use Notebooks?

**Python Scripts (.py)** = Automation
- Run daily to collect data
- Train models on schedule
- Generate predictions automatically

**Jupyter Notebooks (.ipynb)** = Exploration & Learning
- See results instantly
- Visualize data
- Experiment interactively
- Debug problems
- Learn how the system works

## 🎯 Your Notebooks

### **Already Created:**
- `data_fetch.ipynb` - Your original odds fetching notebook (already works!)

### **Should Create:**

#### 1. **Data Exploration Notebook**
```python
# Load data
odds = pd.read_csv('../data/odds_flat.csv')
player_stats = pd.read_csv('../data/player_stats.csv')

# Visualize
odds['spread_home'].hist()
player_stats[['PTS', 'REB', 'AST']].plot(kind='scatter')

# Find patterns
top_scorers = player_stats.nlargest(10, 'PTS')
```

**Use it to:**
- Explore what data looks like
- Find correlations
- Spot outliers
- Compare bookmakers

---

#### 2. **Feature Engineering Notebook**
```python
from feature_engineering.feature_builder import SportsFeatureBuilder

builder = SportsFeatureBuilder()
features = builder.build_player_features(player_stats)

# See what features were created
print(features.columns)

# Check correlations
features.corr()['PTS'].sort_values()
```

**Use it to:**
- Test different feature combinations
- See which features matter most
- Debug feature creation issues

---

#### 3. **Model Training Notebook**
```python
from models.baseline_models import PlayerPropModel

model = PlayerPropModel()
metrics = model.train(X_train, y_train)

# See results immediately
print(f"R²: {metrics['val_r2']}")

# Plot predictions
plt.scatter(y_actual, y_pred)
```

**Use it to:**
- Train models step-by-step
- Visualize predictions vs actual
- Compare different algorithms
- Tune hyperparameters

---

#### 4. **Backtesting Notebook**
```python
from backtesting.backtester import Backtester, ConservativeStrategy

strategy = ConservativeStrategy(initial_bankroll=10000)
backtester = Backtester(strategy)

# Run backtest
results = backtester.run_backtest(historical_data, predictions, odds)

# Visualize performance
backtester.plot_performance()
```

**Use it to:**
- Test betting strategies visually
- See profit curves
- Analyze winning streaks
- Optimize bet sizing

---

## 🚀 Quick Start with Notebooks

1. **Open Jupyter Lab:**
```bash
cd /Users/pranavsingh/sportsbet-ml
source venv/bin/activate
jupyter lab
```

2. **Create a new notebook** in `notebooks/` folder

3. **Start exploring:**
```python
# Cell 1
import pandas as pd
odds = pd.read_csv('../data/odds_flat.csv')
odds.head()

# Cell 2
odds['bookmaker'].value_counts()

# Cell 3
odds['spread_home'].hist()
```

---

## 💡 Notebook Best Practices

### ✅ Good Uses:
- "Let me see what this data looks like"
- "Does my model actually work?"
- "What if I try a different feature?"
- "Show me the profit curve"

### ❌ Not Good For:
- Running daily (use .py scripts instead)
- Production deployment
- Scheduled tasks
- Large-scale processing

---

## 📝 Example: Finding a Betting Edge

**In a notebook:**
```python
# Cell 1: Load data
odds = pd.read_csv('../data/current_odds.csv')
predictions = pd.read_csv('../results/predictions_today.csv')

# Cell 2: Merge predictions with odds
df = predictions.merge(
    odds[odds['market'] == 'player_props'],
    on='player_id'
)

# Cell 3: Find edges
df['edge'] = df['prediction'] - df['line']
edges = df[abs(df['edge']) > 2]  # 2+ point difference

# Cell 4: Visualize
edges.plot(x='player', y='edge', kind='bar')
plt.title('Biggest Betting Edges Tonight')

# Cell 5: Show opportunities
print(edges[['player', 'stat', 'prediction', 'line', 'edge']])
```

**You see results instantly** → Make decision → Place bet

---

## 🎓 Learning Workflow

1. **Start with notebooks** to understand the data
2. **Experiment** with features and models
3. **Once it works**, convert to .py script for automation
4. **Come back to notebooks** when you need to debug

---

## 📦 Ready-to-Use Notebook Templates

I've started creating these in your `notebooks/` folder:
- `01_data_exploration.ipynb` - Load and visualize data
- `02_model_training.ipynb` - Train models interactively
- Your existing `data_fetch.ipynb` - Fetch odds (already works!)

**To use them:**
```bash
jupyter lab notebooks/
```

Then open any `.ipynb` file and click "Run All" or run cell-by-cell.

---

## 🆚 When to Use What

| Task | Use | Why |
|------|-----|-----|
| "Show me the data" | Notebook | Visual, interactive |
| "Does this model work?" | Notebook | See results immediately |
| "Collect data daily at 3pm" | Script | Automation |
| "Train models after data collection" | Script | Automation |
| "I want to try something" | Notebook | Experimentation |
| "Production betting system" | Script | Reliability |

---

## 🎯 Your Next Steps

1. **Open Jupyter:** `jupyter lab`
2. **Create new notebook** in `notebooks/` folder
3. **Name it:** `my_first_exploration.ipynb`
4. **Start with:**
```python
import pandas as pd
odds = pd.read_csv('../data/odds_flat.csv')
odds.info()
odds.describe()
```

5. **Explore!** Add cells, run them, see results

**The beauty of notebooks:** You learn by doing, instantly.
