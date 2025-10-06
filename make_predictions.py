#!/usr/bin/env python3
"""
Make predictions on today's games using trained models.
"""

import sys
import pandas as pd
import numpy as np

sys.path.append('src')

from models.baseline_models import PlayerPropModel
from feature_engineering.feature_builder import SportsFeatureBuilder

def make_todays_predictions():
    """Generate predictions for today's props."""
    
    print("="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)
    
    # Load current data
    print("\n1. Loading current data...")
    try:
        player_stats = pd.read_csv('data/player_stats.csv')
        print(f"   ✓ Loaded {len(player_stats)} players")
    except FileNotFoundError:
        print("   ❌ Data not found! Run 'python3 collect_training_data.py' first.")
        return
    
    # Build features
    print("\n2. Building features...")
    feature_builder = SportsFeatureBuilder()
    features = feature_builder.build_player_features(player_stats)
    
    # Load trained models
    print("\n3. Loading trained models...")
    models = {}
    for stat in ['pts', 'reb', 'ast']:
        try:
            model = PlayerPropModel()
            model.load_model(f'models/{stat}_model.pkl')
            models[stat] = model
            print(f"   ✓ Loaded {stat.upper()} model")
        except FileNotFoundError:
            print(f"   ⚠ No model found for {stat.upper()}")
    
    if not models:
        print("\n❌ No trained models available!")
        print("   Run 'python3 train_models.py' first.")
        return
    
    # Make predictions
    print("\n4. Making predictions for top players...")
    
    # Get top 20 players by points
    top_players = player_stats.nlargest(20, 'PTS')[['PLAYER_NAME', 'PTS', 'REB', 'AST']]
    
    predictions = []
    
    for idx in top_players.index:
        player_name = player_stats.loc[idx, 'PLAYER_NAME']
        
        # Get features for this player
        player_features = features.loc[[idx]].select_dtypes(include=[np.number])
        
        # Remove target columns
        player_features = player_features.drop(columns=['PTS', 'REB', 'AST'], errors='ignore')
        
        # Make predictions with each model
        pred_dict = {'player': player_name}
        
        for stat, model in models.items():
            try:
                pred = model.predict(player_features)[0]
                actual = player_stats.loc[idx, stat.upper()]
                pred_dict[f'pred_{stat}'] = round(pred, 1)
                pred_dict[f'avg_{stat}'] = round(actual, 1)
            except Exception as e:
                pred_dict[f'pred_{stat}'] = None
        
        predictions.append(pred_dict)
    
    # Display predictions
    df_predictions = pd.DataFrame(predictions)
    
    print("\n" + "="*60)
    print("TODAY'S PREDICTIONS")
    print("="*60)
    print(df_predictions.to_string(index=False))
    
    # Save predictions
    df_predictions.to_csv('results/predictions_today.csv', index=False)
    print(f"\n✓ Predictions saved to results/predictions_today.csv")
    
    # Find betting opportunities (where prediction differs significantly)
    print("\n" + "="*60)
    print("BETTING OPPORTUNITIES")
    print("="*60)
    
    opportunities = []
    for _, row in df_predictions.iterrows():
        for stat in ['pts', 'reb', 'ast']:
            pred_col = f'pred_{stat}'
            avg_col = f'avg_{stat}'
            
            if pred_col in row and avg_col in row and pd.notna(row[pred_col]):
                diff = row[pred_col] - row[avg_col]
                
                # If prediction differs by 10%+, it's an opportunity
                if abs(diff) / row[avg_col] > 0.10:
                    opportunities.append({
                        'player': row['player'],
                        'stat': stat.upper(),
                        'prediction': row[pred_col],
                        'season_avg': row[avg_col],
                        'difference': round(diff, 1),
                        'recommendation': 'OVER' if diff > 0 else 'UNDER'
                    })
    
    if opportunities:
        df_opps = pd.DataFrame(opportunities)
        df_opps = df_opps.sort_values('difference', key=abs, ascending=False)
        print(df_opps.to_string(index=False))
        print(f"\n✓ Found {len(opportunities)} potential betting opportunities!")
    else:
        print("No significant betting opportunities found.")
        print("(Predictions are close to season averages)")
    
    return df_predictions, opportunities


if __name__ == "__main__":
    try:
        predictions, opportunities = make_todays_predictions()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

