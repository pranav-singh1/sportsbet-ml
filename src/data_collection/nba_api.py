"""
NBA API data collection module for fetching player and team statistics.
"""

import pandas as pd
import requests
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBAPlayerStats:
    """Client for fetching NBA player statistics and game data."""
    
    def __init__(self):
        self.base_url = "https://stats.nba.com/stats"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.nba.com/',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_player_game_logs(
        self, 
        player_id: str, 
        season: str = "2024-25",
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """Get game logs for a specific player."""
        url = f"{self.base_url}/playergamelog"
        params = {
            'PlayerID': player_id,
            'Season': season,
            'SeasonType': season_type
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'resultSets' in data and len(data['resultSets']) > 0:
                headers = data['resultSets'][0]['headers']
                rows = data['resultSets'][0]['rowSet']
                return pd.DataFrame(rows, columns=headers)
            else:
                logger.warning(f"No data found for player {player_id}")
                return pd.DataFrame()
                
        except requests.RequestException as e:
            logger.error(f"Error fetching player game logs: {e}")
            return pd.DataFrame()
    
    def get_team_game_logs(
        self, 
        team_id: str, 
        season: str = "2024-25",
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """Get game logs for a specific team."""
        url = f"{self.base_url}/teamgamelog"
        params = {
            'TeamID': team_id,
            'Season': season,
            'SeasonType': season_type
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'resultSets' in data and len(data['resultSets']) > 0:
                headers = data['resultSets'][0]['headers']
                rows = data['resultSets'][0]['rowSet']
                return pd.DataFrame(rows, columns=headers)
            else:
                logger.warning(f"No data found for team {team_id}")
                return pd.DataFrame()
                
        except requests.RequestException as e:
            logger.error(f"Error fetching team game logs: {e}")
            return pd.DataFrame()
    
    def get_player_season_stats(
        self, 
        season: str = "2024-25",
        season_type: str = "Regular Season",
        per_mode: str = "PerGame"
    ) -> pd.DataFrame:
        """Get season statistics for all players."""
        url = f"{self.base_url}/leaguedashplayerstats"
        params = {
            'Season': season,
            'SeasonType': season_type,
            'PerMode': per_mode,
            'MeasureType': 'Base',
            'PlusMinus': 'N',
            'PaceAdjust': 'N',
            'Rank': 'N',
            'LeagueID': '00',
            'Outcome': '',
            'Location': '',
            'Month': '0',
            'SeasonSegment': '',
            'DateFrom': '',
            'DateTo': '',
            'OpponentTeamID': '0',
            'VsConference': '',
            'VsDivision': '',
            'GameSegment': '',
            'Period': '0',
            'LastNGames': '0',
            'GameScope': '',
            'PlayerExperience': '',
            'PlayerPosition': '',
            'StarterBench': '',
            'DraftYear': '',
            'DraftPick': '',
            'College': '',
            'Country': '',
            'Height': '',
            'Weight': ''
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'resultSets' in data and len(data['resultSets']) > 0:
                headers = data['resultSets'][0]['headers']
                rows = data['resultSets'][0]['rowSet']
                return pd.DataFrame(rows, columns=headers)
            else:
                logger.warning("No player season stats found")
                return pd.DataFrame()
                
        except requests.RequestException as e:
            logger.error(f"Error fetching player season stats: {e}")
            return pd.DataFrame()
    
    def get_team_season_stats(
        self, 
        season: str = "2024-25",
        season_type: str = "Regular Season",
        per_mode: str = "PerGame"
    ) -> pd.DataFrame:
        """Get season statistics for all teams."""
        url = f"{self.base_url}/leaguedashteamstats"
        params = {
            'Season': season,
            'SeasonType': season_type,
            'PerMode': per_mode,
            'MeasureType': 'Base',
            'PlusMinus': 'N',
            'PaceAdjust': 'N',
            'Rank': 'N',
            'LeagueID': '00',
            'Outcome': '',
            'Location': '',
            'Month': '0',
            'SeasonSegment': '',
            'DateFrom': '',
            'DateTo': '',
            'OpponentTeamID': '0',
            'VsConference': '',
            'VsDivision': '',
            'GameSegment': '',
            'Period': '0',
            'LastNGames': '0',
            'GameScope': '',
            'PlayerExperience': '',
            'PlayerPosition': '',
            'StarterBench': '',
            'DraftYear': '',
            'DraftPick': '',
            'College': '',
            'Country': '',
            'Height': '',
            'Weight': ''
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'resultSets' in data and len(data['resultSets']) > 0:
                headers = data['resultSets'][0]['headers']
                rows = data['resultSets'][0]['rowSet']
                return pd.DataFrame(rows, columns=headers)
            else:
                logger.warning("No team season stats found")
                return pd.DataFrame()
                
        except requests.RequestException as e:
            logger.error(f"Error fetching team season stats: {e}")
            return pd.DataFrame()
    
    def get_schedule(
        self, 
        season: str = "2024-25",
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """Get NBA schedule for the season."""
        url = f"{self.base_url}/schedule"
        params = {
            'Season': season,
            'SeasonType': season_type,
            'LeagueID': '00'
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'resultSets' in data and len(data['resultSets']) > 0:
                headers = data['resultSets'][0]['headers']
                rows = data['resultSets'][0]['rowSet']
                return pd.DataFrame(rows, columns=headers)
            else:
                logger.warning("No schedule data found")
                return pd.DataFrame()
                
        except requests.RequestException as e:
            logger.error(f"Error fetching schedule: {e}")
            return pd.DataFrame()
    
    def get_player_prop_bets_data(
        self, 
        player_id: str, 
        season: str = "2024-25",
        last_n_games: int = 10
    ) -> Dict:
        """Get comprehensive data for player prop betting analysis."""
        # Get recent game logs
        game_logs = self.get_player_game_logs(player_id, season)
        
        if game_logs.empty:
            return {}
        
        # Calculate rolling averages for key stats
        recent_games = game_logs.head(last_n_games)
        
        prop_data = {
            'player_id': player_id,
            'games_analyzed': len(recent_games),
            'avg_points': recent_games['PTS'].mean() if 'PTS' in recent_games.columns else 0,
            'avg_rebounds': recent_games['REB'].mean() if 'REB' in recent_games.columns else 0,
            'avg_assists': recent_games['AST'].mean() if 'AST' in recent_games.columns else 0,
            'avg_steals': recent_games['STL'].mean() if 'STL' in recent_games.columns else 0,
            'avg_blocks': recent_games['BLK'].mean() if 'BLK' in recent_games.columns else 0,
            'avg_turnovers': recent_games['TOV'].mean() if 'TOV' in recent_games.columns else 0,
            'avg_minutes': recent_games['MIN'].mean() if 'MIN' in recent_games.columns else 0,
            'consistency_score': self._calculate_consistency_score(recent_games),
            'trend_direction': self._calculate_trend_direction(recent_games),
            'last_updated': datetime.now().isoformat()
        }
        
        return prop_data
    
    def _calculate_consistency_score(self, game_logs: pd.DataFrame) -> float:
        """Calculate consistency score based on coefficient of variation."""
        if 'PTS' not in game_logs.columns or len(game_logs) < 3:
            return 0.0
        
        points = game_logs['PTS'].values
        if points.std() == 0:
            return 1.0
        
        cv = points.std() / points.mean()
        # Convert to 0-1 scale where 1 is most consistent
        consistency = max(0, 1 - cv)
        return round(consistency, 3)
    
    def _calculate_trend_direction(self, game_logs: pd.DataFrame) -> str:
        """Calculate trend direction for recent performance."""
        if 'PTS' not in game_logs.columns or len(game_logs) < 3:
            return "neutral"
        
        points = game_logs['PTS'].values
        if len(points) >= 3:
            recent_avg = points[:3].mean()
            earlier_avg = points[3:].mean() if len(points) > 3 else points.mean()
            
            if recent_avg > earlier_avg * 1.1:
                return "upward"
            elif recent_avg < earlier_avg * 0.9:
                return "downward"
        
        return "neutral"


def main():
    """Example usage of NBAPlayerStats."""
    nba_client = NBAPlayerStats()
    
    # Get current season player stats
    player_stats = nba_client.get_player_season_stats()
    if not player_stats.empty:
        print(f"Retrieved stats for {len(player_stats)} players")
        print(player_stats[['PLAYER_NAME', 'PTS', 'REB', 'AST']].head())
    
    # Get schedule
    schedule = nba_client.get_schedule()
    if not schedule.empty:
        print(f"Retrieved {len(schedule)} scheduled games")
        print(schedule[['GAME_DATE', 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME']].head())


if __name__ == "__main__":
    main()
