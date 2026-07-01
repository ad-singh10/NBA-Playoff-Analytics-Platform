import pandas as pd


def add_last10(final_df, game_history):
    """
    Adds:
        HOME_LAST10_WIN_PCT
        AWAY_LAST10_WIN_PCT
        LAST10_DIFF
    """

    # Work on a copy
    game_history = game_history.copy()

    # -------------------------------------------------------
    # Create WIN column
    # -------------------------------------------------------

    game_history["WIN"] = (
        game_history["WL"] == "W"
    ).astype(int)

    # -------------------------------------------------------
    # Rolling Last 10 Win %
    # -------------------------------------------------------

    game_history["LAST10_WIN_PCT"] = (
        game_history
        .groupby("TEAM_ID")["WIN"]
        .transform(
            lambda x:
            x.shift(1)
             .rolling(
                 window=10,
                 min_periods=1
             )
             .mean()
        )
    )

    # -------------------------------------------------------
    # Keep required columns
    # -------------------------------------------------------

    last10 = game_history[
        [
            "TEAM_ID",
            "GAME_ID",
            "LAST10_WIN_PCT"
        ]
    ]

    # -------------------------------------------------------
    # Home Team
    # -------------------------------------------------------

    home_last10 = last10.rename(
        columns={
            "TEAM_ID": "HOME_TEAM_ID",
            "LAST10_WIN_PCT": "HOME_LAST10_WIN_PCT"
        }
    )

    final_df = final_df.merge(
        home_last10,
        on=[
            "GAME_ID",
            "HOME_TEAM_ID"
        ],
        how="left"
    )

    # -------------------------------------------------------
    # Away Team
    # -------------------------------------------------------

    away_last10 = last10.rename(
        columns={
            "TEAM_ID": "AWAY_TEAM_ID",
            "LAST10_WIN_PCT": "AWAY_LAST10_WIN_PCT"
        }
    )

    final_df = final_df.merge(
        away_last10,
        on=[
            "GAME_ID",
            "AWAY_TEAM_ID"
        ],
        how="left"
    )

    # -------------------------------------------------------
    # Difference Feature
    # -------------------------------------------------------

    final_df["LAST10_DIFF"] = (
        final_df["HOME_LAST10_WIN_PCT"]
        -
        final_df["AWAY_LAST10_WIN_PCT"]
    )

    return final_df