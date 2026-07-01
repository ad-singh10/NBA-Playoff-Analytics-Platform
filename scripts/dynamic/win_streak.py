import pandas as pd


def add_win_streak(final_df, game_history):
    """
    Adds:
        HOME_WIN_STREAK
        AWAY_WIN_STREAK
        STREAK_DIFF
    """

    game_history = game_history.copy()

    # ----------------------------------------------------
    # Create WIN column
    # ----------------------------------------------------

    game_history["WIN"] = (
        game_history["WL"] == "W"
    ).astype(int)

    # ----------------------------------------------------
    # Calculate Win Streak
    # ----------------------------------------------------

    streaks = []

    for team_id, team_games in game_history.groupby("TEAM_ID"):

        streak = 0

        for _, row in team_games.iterrows():

            # Store streak BEFORE current game
            streaks.append(streak)

            if row["WIN"] == 1:
                streak += 1
            else:
                streak = 0

    game_history["WIN_STREAK"] = streaks

    # ----------------------------------------------------
    # Keep Required Columns
    # ----------------------------------------------------

    streak_df = game_history[
        [
            "TEAM_ID",
            "GAME_ID",
            "WIN_STREAK"
        ]
    ]

    # ----------------------------------------------------
    # Home Team
    # ----------------------------------------------------

    home_streak = streak_df.rename(
        columns={
            "TEAM_ID": "HOME_TEAM_ID",
            "WIN_STREAK": "HOME_WIN_STREAK"
        }
    )

    final_df = final_df.merge(
        home_streak,
        on=[
            "GAME_ID",
            "HOME_TEAM_ID"
        ],
        how="left"
    )

    # ----------------------------------------------------
    # Away Team
    # ----------------------------------------------------

    away_streak = streak_df.rename(
        columns={
            "TEAM_ID": "AWAY_TEAM_ID",
            "WIN_STREAK": "AWAY_WIN_STREAK"
        }
    )

    final_df = final_df.merge(
        away_streak,
        on=[
            "GAME_ID",
            "AWAY_TEAM_ID"
        ],
        how="left"
    )

    # ----------------------------------------------------
    # Difference Feature
    # ----------------------------------------------------

    final_df["STREAK_DIFF"] = (
        final_df["HOME_WIN_STREAK"]
        -
        final_df["AWAY_WIN_STREAK"]
    )

    return final_df