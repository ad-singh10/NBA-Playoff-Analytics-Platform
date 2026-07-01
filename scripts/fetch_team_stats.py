import pandas as pd
from nba_api.stats.endpoints import LeagueDashTeamStats
from time import sleep

# Seasons to Download
seasons = [
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25"
]

# List to Store Data
all_team_stats = []

# Download Data
for season in seasons:

    print(f"Downloading Team Stats: {season}")

    team_stats = LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Advanced"
    )

    # Convert API response to DataFrame
    df = team_stats.get_data_frames()[0]

    # Add season column
    df["SEASON"] = season

    # Store DataFrame
    all_team_stats.append(df)

    # Wait 1 second
    sleep(1)

# Combine all seasons
team_stats = pd.concat(
    all_team_stats,
    ignore_index=True
)

print("\nCombined Team Stats Shape:")
print(team_stats.shape)

# Save CSV
team_stats.to_csv(
    "data/raw/team_stats.csv",
    index=False
)

print("\nTeam statistics saved successfully!")