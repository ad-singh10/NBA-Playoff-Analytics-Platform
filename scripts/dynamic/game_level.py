import pandas as pd


def build_game_level(game_history):
    """
    Converts team-wise game history into one row per game.
    """

    games = []

    # Process every game
    for game_id, game in game_history.groupby("GAME_ID"):

        # Every NBA game should have exactly two rows
        if len(game) != 2:
            continue

        # Identify home & away teams
        home = game[game["MATCHUP"].str.contains("vs.")]
        away = game[game["MATCHUP"].str.contains("@")]

        if home.empty or away.empty:
            continue

        home = home.iloc[0]
        away = away.iloc[0]

        games.append(
            {
                "GAME_ID": game_id,

                "GAME_DATE": home["GAME_DATE"],

                "HOME_TEAM_ID": home["TEAM_ID"],
                "AWAY_TEAM_ID": away["TEAM_ID"],

                "HOME_TEAM": home["TEAM_NAME"],
                "AWAY_TEAM": away["TEAM_NAME"],

                "HOME_WIN": 1 if home["WL"] == "W" else 0,

                "HOME_POINTS": home["PTS"],
                "AWAY_POINTS": away["PTS"],

                "MARGIN": abs(home["PLUS_MINUS"])
            }
        )

    game_level = pd.DataFrame(games)

    game_level = game_level.sort_values(
        "GAME_DATE"
    ).reset_index(drop=True)

    return game_level