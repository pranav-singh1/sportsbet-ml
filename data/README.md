# Data Directory

This directory contains the data files for the Sports Betting ML System.

## Structure

- `raw_odds.csv` - Raw odds data from The Odds API
- `odds_flat.csv` - Flattened and processed odds data
- `sample_*.csv` - Sample data files for testing (these are tracked in git)

## Data Sources

1. **The Odds API** - Sports betting odds from various bookmakers
2. **NBA API** - Player and team statistics
3. **ESPN API** - Game results and schedules
4. **Custom scrapers** - Additional data sources

## Data Collection

Run the data collection scripts to fetch fresh data:

```bash
# Fetch NBA odds
python src/data_collection/odds_api.py

# Fetch NBA player stats
python src/data_collection/nba_api.py
```

## Data Format

### Odds Data
- `game_id` - Unique identifier for each game
- `home_team` - Home team name
- `away_team` - Away team name
- `bookmaker` - Sportsbook name
- `market` - Betting market (h2h, spreads, totals)
- `odds_*` - Odds in various formats
- `commence_time` - Game start time
- `last_update` - Last odds update time

### Player Stats
- `PLAYER_ID` - Unique player identifier
- `PLAYER_NAME` - Player name
- `PTS` - Points per game
- `REB` - Rebounds per game
- `AST` - Assists per game
- `STL` - Steals per game
- `BLK` - Blocks per game
- `TOV` - Turnovers per game
- `MIN` - Minutes per game

## Data Quality

- All data is validated for completeness and accuracy
- Missing values are handled appropriately
- Outliers are detected and flagged
- Data freshness is monitored

## Privacy and Security

- No personal information is stored
- All data is publicly available sports statistics
- API keys are stored securely and not committed to git
