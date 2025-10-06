"""
OddsAPI data collection module for fetching sports betting odds.
"""

import requests
import pandas as pd
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OddsAPIClient:
    """Client for fetching odds data from The Odds API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.session = requests.Session()
        
    def get_sports(self) -> List[Dict]:
        """Get list of available sports."""
        url = f"{self.base_url}/sports"
        params = {"apiKey": self.api_key}
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching sports: {e}")
            return []
    
    def get_odds(
        self, 
        sport: str, 
        regions: List[str] = ["us"], 
        markets: List[str] = ["h2h", "spreads", "totals"],
        odds_format: str = "american",
        date_format: str = "iso"
    ) -> List[Dict]:
        """Fetch odds for a specific sport."""
        url = f"{self.base_url}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": ",".join(regions),
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
            "dateFormat": date_format
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching odds for {sport}: {e}")
            return []
    
    def get_historical_odds(
        self, 
        sport: str, 
        regions: List[str] = ["us"],
        markets: List[str] = ["h2h", "spreads", "totals"],
        odds_format: str = "american",
        date_format: str = "iso",
        days_back: int = 7
    ) -> List[Dict]:
        """Fetch historical odds for the past N days."""
        all_odds = []
        
        for i in range(days_back):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            url = f"{self.base_url}/sports/{sport}/odds-history"
            params = {
                "apiKey": self.api_key,
                "regions": ",".join(regions),
                "markets": ",".join(markets),
                "oddsFormat": odds_format,
                "dateFormat": date_format,
                "commenceTimeFrom": date,
                "commenceTimeTo": (datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ") + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                odds_data = response.json()
                all_odds.extend(odds_data)
                time.sleep(0.1)  # Rate limiting
            except requests.RequestException as e:
                logger.error(f"Error fetching historical odds for {date}: {e}")
                
        return all_odds
    
    def flatten_odds_data(self, odds_data: List[Dict]) -> pd.DataFrame:
        """Convert nested odds data to flat DataFrame."""
        rows = []
        
        for game in odds_data:
            game_id = game["id"]
            home_team = game["home_team"]
            away_team = game["away_team"]
            commence_time = game["commence_time"]
            
            for bookmaker in game.get("bookmakers", []):
                book_name = bookmaker["title"]
                last_update = bookmaker["last_update"]
                
                for market in bookmaker.get("markets", []):
                    market_key = market["key"]
                    outcomes = market["outcomes"]
                    
                    if market_key == "spreads":
                        spread_home = None
                        spread_away = None
                        price_home = None
                        price_away = None
                        
                        for outcome in outcomes:
                            if outcome["name"] == home_team:
                                spread_home = outcome["point"]
                                price_home = outcome["price"]
                            else:
                                spread_away = outcome["point"]
                                price_away = outcome["price"]
                        
                        rows.append({
                            "game_id": game_id,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": book_name,
                            "market": market_key,
                            "spread_home": spread_home,
                            "spread_away": spread_away,
                            "price_home": price_home,
                            "price_away": price_away,
                            "commence_time": commence_time,
                            "last_update": last_update
                        })
                    
                    elif market_key == "totals":
                        total_over = None
                        total_under = None
                        price_over = None
                        price_under = None
                        
                        for outcome in outcomes:
                            if outcome["name"] == "Over":
                                total_over = outcome["point"]
                                price_over = outcome["price"]
                            else:
                                total_under = outcome["point"]
                                price_under = outcome["price"]
                        
                        rows.append({
                            "game_id": game_id,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": book_name,
                            "market": market_key,
                            "total_over": total_over,
                            "total_under": total_under,
                            "price_over": price_over,
                            "price_under": price_under,
                            "commence_time": commence_time,
                            "last_update": last_update
                        })
                    
                    elif market_key == "h2h":
                        price_home = None
                        price_away = None
                        
                        for outcome in outcomes:
                            if outcome["name"] == home_team:
                                price_home = outcome["price"]
                            else:
                                price_away = outcome["price"]
                        
                        rows.append({
                            "game_id": game_id,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": book_name,
                            "market": market_key,
                            "price_home": price_home,
                            "price_away": price_away,
                            "commence_time": commence_time,
                            "last_update": last_update
                        })
        
        return pd.DataFrame(rows)
    
    def save_odds_to_csv(self, odds_data: List[Dict], filename: str):
        """Save odds data to CSV file."""
        df = self.flatten_odds_data(odds_data)
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(df)} odds records to {filename}")
        return df


def main():
    """Example usage of OddsAPIClient."""
    # Initialize client with your API key
    api_key = "34ed2dc9566ad15743e1ef7eac40a2ca"  # Replace with your actual API key
    client = OddsAPIClient(api_key)
    
    # Fetch current NBA odds
    nba_odds = client.get_odds("basketball_nba")
    
    if nba_odds:
        # Save to CSV
        df = client.save_odds_to_csv(nba_odds, "../data/nba_odds_current.csv")
        print(f"Fetched {len(df)} odds records")
        print(df.head())
    else:
        print("No odds data retrieved")


if __name__ == "__main__":
    main()
