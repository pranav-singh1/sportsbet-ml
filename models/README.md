# Models Directory

This directory contains trained machine learning models for sports betting predictions.

## Model Types

### Baseline Models
- **Random Forest** - Ensemble of decision trees for robust predictions
- **XGBoost** - Gradient boosting for high-performance predictions
- **Linear Regression** - Simple baseline for comparison
- **Gradient Boosting** - Another ensemble method

### Advanced Models (Coming Soon)
- **LSTM** - Recurrent neural networks for time-series data
- **Transformer** - Attention-based models for sequence prediction
- **Ensemble** - Combination of multiple models

## Model Files

- `*.pkl` - Pickled model files (not tracked in git)
- `*.joblib` - Joblib serialized models (not tracked in git)
- `*.h5` - Keras/TensorFlow models (not tracked in git)
- `*.pt` - PyTorch models (not tracked in git)

## Model Training

Train models using the main pipeline:

```bash
# Train all models
python src/main.py

# Train specific model
python src/models/baseline_models.py
```

## Model Performance

Models are evaluated on:
- **Accuracy** - Percentage of correct predictions
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **F1 Score** - Harmonic mean of precision and recall
- **ROI** - Return on investment in backtesting
- **Sharpe Ratio** - Risk-adjusted returns

## Model Deployment

Models can be deployed via:
- **FastAPI** - REST API for real-time predictions
- **Batch Processing** - Scheduled predictions
- **Streaming** - Real-time data processing

## Model Monitoring

- Performance tracking over time
- Drift detection
- Automatic retraining triggers
- A/B testing capabilities

## Best Practices

1. **Version Control** - Track model versions and performance
2. **Validation** - Always validate on unseen data
3. **Monitoring** - Monitor model performance in production
4. **Retraining** - Regular retraining with fresh data
5. **Documentation** - Document model assumptions and limitations
