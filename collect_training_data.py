#!/usr/bin/env python3
"""
Collect training data: odds + player stats + game results
This creates the dataset needed to train profitable models.
"""

import sys
import os
import pandas as pd
import time
from datetime import datetime

sys.path.append('src')

from data_collection.odds_api import OddsAPIClient
from data_collection.nba_api import NBAPlayerStats
from config.config import ODDS_API_KEY

def collect_full_dataset():
    """Collect comprehensive training data."""
    
    print("="*60)
    print("COLLECTING TRAINING DATA")
    print("="*60)
    
    # Initialize clients
    odds_client = OddsAPIClient(ODDS_API_KEY)
    nba_client = NBAPlayerStats()
    
    # 1. Get current NBA odds
    print("\n1. Fetching current NBA odds...")
    odds = odds_client.get_odds('basketball_nba')
    df_odds = odds_client.flatten_odds_data(odds)
    print(f"   ✓ {len(df_odds)} odds records collected")
    
    # Save
    df_odds.to_csv('data/current_odds.csv', index=False)
    print(f"   ✓ Saved to data/current_odds.csv")
    
    # 2. Get player season stats
    print("\n2. Fetching NBA player stats...")
    time.sleep(1)
    player_stats = nba_client.get_player_season_stats()
    print(f"   ✓ {len(player_stats)} players collected")
    
    # Save
    player_stats.to_csv('data/player_stats.csv', index=False)
    print(f"   ✓ Saved to data/player_stats.csv")
    
    # 3. Get team stats
    print("\n3. Fetching NBA team stats...")
    time.sleep(1)
    team_stats = nba_client.get_team_season_stats()
    print(f"   ✓ {len(team_stats)} teams collected")
    
    # Save
    team_stats.to_csv('data/team_stats.csv', index=False)
    print(f"   ✓ Saved to data/team_stats.csv")
    
    # 4. Get recent game logs for top players
    print("\n4. Fetching recent game logs (this takes a while)...")
    top_players = player_stats.nlargest(10, 'PTS')[['PLAYER_ID', 'PLAYER_NAME', 'PTS']]
    
    all_game_logs = []
    for idx, row in top_players.iterrows():
        player_id = row['PLAYER_ID']
        player_name = row['PLAYER_NAME']
        
        print(f"   Fetching {player_name}...")
        time.sleep(0.6)  # Rate limiting
        
        game_logs = nba_client.get_player_game_logs(str(player_id))
        if not game_logs.empty:
            game_logs['PLAYER_NAME'] = player_name
            all_game_logs.append(game_logs)
    
    if all_game_logs:
        df_game_logs = pd.concat(all_game_logs, ignore_index=True)
        df_game_logs.to_csv('data/game_logs.csv', index=False)
        print(f"   ✓ {len(df_game_logs)} game logs saved to data/game_logs.csv")
    
    # Summary
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  - data/current_odds.csv ({len(df_odds)} records)")
    print(f"  - data/player_stats.csv ({len(player_stats)} players)")
    print(f"  - data/team_stats.csv ({len(team_stats)} teams)")
    if all_game_logs:
        print(f"  - data/game_logs.csv ({len(df_game_logs)} games)")
    
    print(f"\n✓ Ready for feature engineering and model training!")
    
    return {
        'odds': df_odds,
        'player_stats': player_stats,
        'team_stats': team_stats,
        'game_logs': df_game_logs if all_game_logs else None
    }


if __name__ == "__main__":
    try:
        data = collect_full_dataset()
        print("\n🎉 Success! Run 'python3 train_models.py' next.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have internet connection and valid API keys.")

