import pandas as pd


def add_rest_days(final_df, game_history):
    """
    Adds:
        HOME_REST_DAYS
        AWAY_REST_DAYS
        REST_DAY_DIFF
    """

    # Sort chronologically
    game_history = game_history.sort_values(
        ["TEAM_ID", "GAME_DATE"]
    ).copy()

    # Previous game for each team
    game_history["PREVIOUS_GAME_DATE"] = (
        game_history
        .groupby("TEAM_ID")["GAME_DATE"]
        .shift(1)
    )

    # Calculate rest days
    game_history["REST_DAYS"] = (
        game_history["GAME_DATE"] -
        game_history["PREVIOUS_GAME_DATE"]
    ).dt.days

    # Keep required columns
    rest_days = game_history[
        [
            "TEAM_ID",
            "GAME_ID",
            "REST_DAYS"
        ]
    ]

    # ---------------- HOME ----------------

    home_rest = rest_days.rename(
        columns={
            "TEAM_ID": "HOME_TEAM_ID",
            "REST_DAYS": "HOME_REST_DAYS"
        }
    )

    final_df = final_df.merge(
        home_rest,
        on=["GAME_ID", "HOME_TEAM_ID"],
        how="left"
    )

    # ---------------- AWAY ----------------

    away_rest = rest_days.rename(
        columns={
            "TEAM_ID": "AWAY_TEAM_ID",
            "REST_DAYS": "AWAY_REST_DAYS"
        }
    )

    final_df = final_df.merge(
        away_rest,
        on=["GAME_ID", "AWAY_TEAM_ID"],
        how="left"
    )

    # Difference Feature
    final_df["REST_DAY_DIFF"] = (
        final_df["HOME_REST_DAYS"] -
        final_df["AWAY_REST_DAYS"]
    )

    return final_df