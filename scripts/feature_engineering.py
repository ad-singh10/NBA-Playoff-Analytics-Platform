import pandas as pd

#read Playoff games

playoff_games = pd.read_csv(
    "data/raw/playoff_games.csv"
)

#Inspect

print(playoff_games.head())

print(playoff_games.shape)

print(playoff_games.columns.tolist())

print(playoff_games["GAME_ID"].value_counts().head())

#Verifying Assumption

print("\nGAME_ID Frequency:")
print(playoff_games["GAME_ID"].value_counts().head(10))

#Inspecting a sample

sample_game = playoff_games[
    playoff_games["GAME_ID"] == playoff_games["GAME_ID"].iloc[0]
]

print("\nSample Game:")
print(sample_game)

#Create one row per playoff game

games_list = []

for game_id,game in playoff_games.groupby("GAME_ID") :

     home_team = game[game["MATCHUP"].str.contains("vs.")]
     away_team = game[game["MATCHUP"].str.contains("@")]

     if home_team.empty or away_team.empty:
          continue
     
     home_team = home_team.iloc[0]
     away_team = away_team.iloc[0]
     
     game_data = {
    "GAME_ID": game_id,
    "SEASON_ID": home_team["SEASON_ID"],
    "GAME_DATE": home_team["GAME_DATE"],
    "HOME_TEAM": home_team["TEAM_NAME"],
    "AWAY_TEAM": away_team["TEAM_NAME"],
    "HOME_TEAM_ID": home_team["TEAM_ID"],
    "AWAY_TEAM_ID": away_team["TEAM_ID"],
    "HOME_WIN": 1 if home_team["WL"] == "W" else 0
}

     games_list.append(game_data)


# Convert list of dictionaries to DataFrame
final_df = pd.DataFrame(games_list)

#Load Team_statistics

team_stats = pd.read_csv(
     "data/raw/team_stats.csv"
)

print("\nFinal Dataset Shape:")
print(final_df.shape)

print("\nFirst 5 Rows:")
print(final_df.head())

# Save the dataset
#final_df.to_csv(
#    "data/processed/final_dataset.csv",
#    index=False
#)

#print("\nFinal dataset saved successfully!")

# Convert season ID to season string

season_map = {
    42015: "2015-16",
    42016: "2016-17",
    42017: "2017-18",
    42018: "2018-19",
    42019: "2019-20",
    42020: "2020-21",
    42021: "2021-22",
    42022: "2022-23",
    42023: "2023-24",
    42024: "2024-25"
}

final_df["SEASON"] = final_df["SEASON_ID"].map(season_map)

print(final_df[["SEASON_ID", "SEASON"]].head())

#Prepare Home team stats

home_stats =  team_stats.drop(
     columns=[
    "TEAM_NAME",
    "MIN",
    "W_RANK",
    "L_RANK",
    "W_PCT_RANK",
    "MIN_RANK",
    "OFF_RATING_RANK",
    "DEF_RATING_RANK",
    "NET_RATING_RANK",
    "AST_PCT_RANK",
    "AST_TO_RANK",
    "AST_RATIO_RANK",
    "OREB_PCT_RANK",
    "DREB_PCT_RANK",
    "REB_PCT_RANK",
    "TM_TOV_PCT_RANK",
    "EFG_PCT_RANK",
    "TS_PCT_RANK",
    "PACE_RANK",
    "PIE_RANK"
]
          
     
)

home_stats = home_stats.rename(
     columns= lambda col :
      "HOME_TEAM_ID" if col == "TEAM_ID"
        else f"HOME_{col}" if col != "SEASON"
        else col
)


#merge home team Stat

final_df = final_df.merge(
     home_stats,
     on = ["HOME_TEAM_ID","SEASON"],
     how="left"

)

print("\nAfter Home Team Merge:")
print(final_df.shape)

print("\nColumns:")
print(final_df.columns.tolist())


#Create Away team stats

away_stats = team_stats.drop(
     columns=[
     "TEAM_NAME",
        "MIN",

        "W_RANK",
        "L_RANK",
        "W_PCT_RANK",
        "MIN_RANK",
        "OFF_RATING_RANK",
        "DEF_RATING_RANK",
        "NET_RATING_RANK",
        "AST_PCT_RANK",
        "AST_TO_RANK",
        "AST_RATIO_RANK",
        "OREB_PCT_RANK",
        "DREB_PCT_RANK",
        "REB_PCT_RANK",
        "TM_TOV_PCT_RANK",
        "EFG_PCT_RANK",
        "TS_PCT_RANK",
        "PACE_RANK",
        "PIE_RANK"     
     ]
)

#Renaming Columns

away_stats = away_stats.rename(
    columns=lambda col:
        "AWAY_TEAM_ID" if col == "TEAM_ID"
        else f"AWAY_{col}" if col != "SEASON"
        else col
)

#Merging the Stats

final_df = final_df.merge(
    away_stats,
    on=["AWAY_TEAM_ID", "SEASON"],
    how="left"
)

print("\nAfter Away Team Merge:")
print(final_df.shape)

print(final_df.head())

#Verifying NUll values

print(final_df[[
    "HOME_TEAM",
    "HOME_OFF_RATING",
    "AWAY_TEAM",
    "AWAY_OFF_RATING"
]].head())

print("\nMissing Values:")
print(final_df.isnull().sum().sort_values(ascending=False).head(10))


# Difference Features

final_df["NET_RATING_DIFF"] = (
    final_df["HOME_NET_RATING"] -
    final_df["AWAY_NET_RATING"]
)

final_df["OFF_RATING_DIFF"] = (
    final_df["HOME_OFF_RATING"] -
    final_df["AWAY_OFF_RATING"]
)

final_df["DEF_RATING_DIFF"] = (
    final_df["HOME_DEF_RATING"] -
    final_df["AWAY_DEF_RATING"]
)

final_df["PACE_DIFF"] = (
    final_df["HOME_PACE"] -
    final_df["AWAY_PACE"]
)

final_df["TS_PCT_DIFF"] = (
    final_df["HOME_TS_PCT"] -
    final_df["AWAY_TS_PCT"]
)

final_df["EFG_PCT_DIFF"] = (
    final_df["HOME_EFG_PCT"] -
    final_df["AWAY_EFG_PCT"]
)

final_df["PIE_DIFF"] = (
    final_df["HOME_PIE"] -
    final_df["AWAY_PIE"]
)

print("\nDifference Features:")

print(final_df[
    [
        "HOME_TEAM",
            "AWAY_TEAM",

            "NET_RATING_DIFF",
            "OFF_RATING_DIFF",
            "DEF_RATING_DIFF",
            "PACE_DIFF",
            "TS_PCT_DIFF",
            "EFG_PCT_DIFF",
            "PIE_DIFF"
    ]
].head())

# Checking NUll values
print("\nMissing Values in Difference Features:")

print(
    final_df[
        [
            "NET_RATING_DIFF",
            "OFF_RATING_DIFF",
            "DEF_RATING_DIFF",
            "PACE_DIFF",
            "TS_PCT_DIFF",
            "EFG_PCT_DIFF",
            "PIE_DIFF"
        ]
    ].isnull().sum()
)

print("\nFinal Dataset Shape:")
print(final_df.shape)

final_df.to_csv(
    "data/processed/final_dataset.csv",
    index=False
)

print("Static feature engineering completed.")






