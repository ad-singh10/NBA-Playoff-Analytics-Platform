import pandas as pd
import numpy as np
import math

INITIAL_ELO = 1500

HOME_ADVANTAGE = 65


def expected_score(team_elo, opponent_elo):

    return 1 / (
        1 + 10 ** ((opponent_elo - team_elo) / 400)
    )


def margin_multiplier(margin):
     return min(math.log(margin + 1), 2.5)


def dynamic_k(elo_difference):

    if elo_difference < 100:
        return 24

    elif elo_difference < 200:
        return 20

    else:
        return 16


def add_elo(final_df, game_level):

    # -----------------------------------------------------
    # Initialize Ratings
    # -----------------------------------------------------

    ratings = {}

    home_elo = []

    away_elo = []

    # -----------------------------------------------------
    # Chronological Order
    # -----------------------------------------------------

    game_level = game_level.sort_values(
        "GAME_DATE"
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Process Every Game
    # -----------------------------------------------------

    for _, game in game_level.iterrows():

        home = game["HOME_TEAM_ID"]

        away = game["AWAY_TEAM_ID"]

        ratings.setdefault(home, INITIAL_ELO)

        ratings.setdefault(away, INITIAL_ELO)

        actual_home = ratings[home]

        actual_away = ratings[away]

        # Store Pregame Ratings

        home_elo.append(actual_home)

        away_elo.append(actual_away)

        # ------------------------------------------
        # Home Court Advantage
        # ------------------------------------------

        home_for_prediction = actual_home + HOME_ADVANTAGE

        away_for_prediction = actual_away

        expected_home = expected_score(
            home_for_prediction,
            away_for_prediction
        )

        expected_away = expected_score(
            away_for_prediction,
            home_for_prediction
        )

        # ------------------------------------------
        # Game Result
        # ------------------------------------------

        home_win = game["HOME_WIN"]

        margin = game["MARGIN"]

        multiplier = margin_multiplier(
            margin
        )

        k = dynamic_k(
            abs(actual_home - actual_away)
        )

        # ------------------------------------------
        # Update Ratings
        # ------------------------------------------

        if home_win == 1:

            ratings[home] = (
                actual_home
                +
                k
                *
                multiplier
                *
                (1 - expected_home)
            )

            ratings[away] = (
                actual_away
                +
                k
                *
                multiplier
                *
                (0 - expected_away)
            )

        else:

            ratings[home] = (
                actual_home
                +
                k
                *
                multiplier
                *
                (0 - expected_home)
            )

            ratings[away] = (
                actual_away
                +
                k
                *
                multiplier
                *
                (1 - expected_away)
            )

    # -----------------------------------------------------
    # Save Pregame Ratings
    # -----------------------------------------------------

    game_level["HOME_ELO"] = home_elo

    game_level["AWAY_ELO"] = away_elo

    game_level["ELO_DIFF"] = (
        game_level["HOME_ELO"]
        -
        game_level["AWAY_ELO"]
    )

        # -----------------------------------------------------
    # Keep only required Elo columns
    # -----------------------------------------------------

    elo_df = game_level[
        [
            "GAME_ID",
            "HOME_TEAM_ID",
            "AWAY_TEAM_ID",
            "HOME_ELO",
            "AWAY_ELO",
            "ELO_DIFF"
        ]
    ]

    # -----------------------------------------------------
    # Merge Home Elo
    # -----------------------------------------------------

    final_df = final_df.merge(
        elo_df[
            [
                "GAME_ID",
                "HOME_TEAM_ID",
                "HOME_ELO"
            ]
        ],
        on=[
            "GAME_ID",
            "HOME_TEAM_ID"
        ],
        how="left"
    )

    # -----------------------------------------------------
    # Merge Away Elo
    # -----------------------------------------------------

    final_df = final_df.merge(
        elo_df[
            [
                "GAME_ID",
                "AWAY_TEAM_ID",
                "AWAY_ELO"
            ]
        ],
        on=[
            "GAME_ID",
            "AWAY_TEAM_ID"
        ],
        how="left"
    )

    # -----------------------------------------------------
    # Merge Elo Difference
    # -----------------------------------------------------

    final_df = final_df.merge(
        elo_df[
            [
                "GAME_ID",
                "ELO_DIFF"
            ]
        ],
        on="GAME_ID",
        how="left"
    )

    print("\n======================================")
    print("ELO FEATURES ADDED")
    print("======================================")

    print(
        final_df[
            [
                "HOME_TEAM",
                "AWAY_TEAM",
                "HOME_ELO",
                "AWAY_ELO",
                "ELO_DIFF"
            ]
        ].head(10)
    )

    print("\nMissing Values")

    print(
        final_df[
            [
                "HOME_ELO",
                "AWAY_ELO",
                "ELO_DIFF"
            ]
        ].isnull().sum()
    )

    return final_df