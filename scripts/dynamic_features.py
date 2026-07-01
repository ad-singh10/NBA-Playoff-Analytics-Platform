import pandas as pd

from dynamic.rest_days import add_rest_days
from dynamic.last10 import add_last10
from dynamic.win_streak import add_win_streak
from dynamic.game_level import build_game_level
from dynamic.elo import add_elo


# ==========================================================
# LOAD DATASETS
# ==========================================================

regular_games = pd.read_csv(
    "data/raw/regular_season_games.csv"
)

playoff_games = pd.read_csv(
    "data/raw/playoff_games.csv"
)

final_df = pd.read_csv(
    "data/processed/final_dataset.csv"
)

# ==========================================================
# CONVERT DATES
# ==========================================================

regular_games["GAME_DATE"] = pd.to_datetime(
    regular_games["GAME_DATE"]
)

playoff_games["GAME_DATE"] = pd.to_datetime(
    playoff_games["GAME_DATE"]
)

final_df["GAME_DATE"] = pd.to_datetime(
    final_df["GAME_DATE"]
)

# ==========================================================
# KEEP REQUIRED COLUMNS
# ==========================================================

regular_games = regular_games[
    [
        "TEAM_ID",
        "TEAM_NAME",
        "GAME_ID",
        "GAME_DATE",
        "MATCHUP",
        "WL",
        "PTS",
        "PLUS_MINUS"
    ]
]

playoff_games = playoff_games[
    [
        "TEAM_ID",
        "TEAM_NAME",
        "GAME_ID",
        "GAME_DATE",
        "MATCHUP",
        "WL",
        "PTS",
        "PLUS_MINUS"
    ]
]

# ==========================================================
# BUILD GAME HISTORY
# ==========================================================

game_history = pd.concat(
    [
        regular_games,
        playoff_games
    ],
    ignore_index=True
)

game_history = game_history.sort_values(
    by=[
        "TEAM_ID",
        "GAME_DATE"
    ]
)

game_level = build_game_level(game_history)

print("\nGame Level Dataset")

print(game_level.head())

print("\nShape")

print(game_level.shape)
# ==========================================================
# REST DAYS
# ==========================================================

final_df = add_rest_days(
    final_df,
    game_history
)

print(" Rest Days Added")


print("\nCurrent Dataset Shape")
print(final_df.shape)

# ==========================================================
# SAVE DATASET
# ==========================================================

final_df.to_csv(
    "data/processed/final_dataset_dynamic.csv",
    index=False
)

print("\nDynamic dataset saved successfully!")

# ==========================================================
# LAST 10 FEATURES
# ==========================================================

final_df = add_last10(
    final_df,
    game_history
)

print("\n======================================")
print("✓ Last 10 Added")
print("======================================")

print(
    final_df[
        [
            "HOME_TEAM",
            "AWAY_TEAM",
            "HOME_LAST10_WIN_PCT",
            "AWAY_LAST10_WIN_PCT",
            "LAST10_DIFF"
        ]
    ].head(10)
)

print("\nMissing Values")

print(
    final_df[
        [
            "HOME_LAST10_WIN_PCT",
            "AWAY_LAST10_WIN_PCT",
            "LAST10_DIFF"
        ]
    ].isnull().sum()
)

print("\nCurrent Dataset Shape")
print(final_df.shape)

# ==========================================================
# WIN STREAK
# ==========================================================

final_df = add_win_streak(
    final_df,
    game_history
)

print(" Win Streak Added")

# ==========================================================
# ENHANCED ELO
# ==========================================================

final_df = add_elo(
    final_df,
    game_level
)

print("✓ Enhanced Elo Added")

print("\nFINAL COLUMNS")
print(final_df.columns.tolist())

print("\nFINAL SHAPE")
print(final_df.shape)

final_df.to_csv(
    "data/processed/final_dataset_dynamic.csv",
    index=False
)

print("\nDynamic Features Saved Successfully!")
print(final_df.shape)