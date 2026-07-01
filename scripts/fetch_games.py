import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder
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

#Lists to store data
all_regular_games=[]
all_playoff_games=[]

#Download data
for season in seasons:

    print(f"Downloading {season}")

    regular_games = LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season"
    )

    regular_df = regular_games.get_data_frames()[0]
    all_regular_games.append(regular_df)

    playoff_games = LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Playoffs"
    )

    playoff_df = playoff_games.get_data_frames()[0]
    all_playoff_games.append(playoff_df)

    sleep(1)

#Combine Data

regular_games = pd.concat(
    all_regular_games,
    ignore_index=True
)

playoff_games =  pd.concat(
    all_playoff_games,
    ignore_index=True
)

#Save CSV Files

regular_games.to_csv(
    "data/raw/regular_season_games.csv",
    index=False
)

playoff_games.to_csv(
    "data/raw/playoff_games.csv",
    index=False
)

#Summary

print("\n==============================")
print("Download Completed Successfully")
print("==============================")

print("\nRegular Season Dataset Shape:")
print(regular_games.shape)

print("\nPlayoff Dataset Shape:")
print(playoff_games.shape)

print("\nFiles Saved:")

print("data/raw/regular_season_games.csv")
print("data/raw/playoff_games.csv")


