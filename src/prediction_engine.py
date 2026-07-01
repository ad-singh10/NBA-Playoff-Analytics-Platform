import joblib
import pandas as pd

# ==========================================================
# LOAD DATASET
# ==========================================================

dataset = pd.read_csv(
    "data/processed/final_dataset_dynamic.csv"
)

# ==========================================================
# FEATURES
# ==========================================================

FEATURES = [

    "NET_RATING_DIFF",
    "OFF_RATING_DIFF",
    "DEF_RATING_DIFF",
    "PACE_DIFF",
    "TS_PCT_DIFF",
    "EFG_PCT_DIFF",
    "PIE_DIFF",

    "HOME_LAST10_WIN_PCT",
    "AWAY_LAST10_WIN_PCT",
    "LAST10_DIFF",

    "HOME_WIN_STREAK",
    "AWAY_WIN_STREAK",
    "STREAK_DIFF",

    "HOME_REST_DAYS",
    "AWAY_REST_DAYS",
    "REST_DAY_DIFF",

    "HOME_ELO",
    "AWAY_ELO",
    "ELO_DIFF"

]

# ==========================================================
# LOAD MODELS
# ==========================================================

logistic_model = joblib.load(
    "models/logistic_regression.pkl"
)

random_forest_model = joblib.load(
    "models/random_forest_tuned.pkl"
)

xgboost_model = joblib.load(
    "models/xgboost_tuned.pkl"
)

# ==========================================================
# LOAD SCALER
# ==========================================================

scaler = joblib.load(
    "models/scaler.pkl"
)

# ==========================================================
# GET MODEL
# ==========================================================

def get_model(model_name):

    if model_name == "Logistic Regression":

        return logistic_model

    elif model_name == "Random Forest":

        return random_forest_model

    else:

        return xgboost_model
    

# ==========================================================
# GET LATEST TEAM STATS
# ==========================================================

def get_latest_team_stats(team_name):

    # Latest game where team was HOME
    home_games = dataset[
        dataset["HOME_TEAM"] == team_name
    ]

    # Latest game where team was AWAY
    away_games = dataset[
        dataset["AWAY_TEAM"] == team_name
    ]

    latest_home = None
    latest_away = None

    if not home_games.empty:

        latest_home = home_games.sort_values(
            "GAME_DATE",
            ascending=False
        ).iloc[0]

    if not away_games.empty:

        latest_away = away_games.sort_values(
            "GAME_DATE",
            ascending=False
        ).iloc[0]

    # ------------------------------------------------------

    if latest_home is None:

        return latest_away

    if latest_away is None:

        return latest_home

    # ------------------------------------------------------

    if latest_home["GAME_DATE"] > latest_away["GAME_DATE"]:

        return latest_home

    return latest_away   

# ==========================================================
# BUILD FEATURE VECTOR
# ==========================================================

def build_feature_vector(home_team, away_team):

    home = get_latest_team_stats(home_team)
    away = get_latest_team_stats(away_team)

    # ==========================================================
    # HOME TEAM STATS
    # ==========================================================

    home_stats = {

        "elo_rating": float(home["HOME_ELO"]),
        "net_rating": float(home["HOME_NET_RATING"]),
        "off_rating": float(home["HOME_OFF_RATING"]),
        "def_rating": float(home["HOME_DEF_RATING"]),
        "pace": float(home["HOME_PACE"]),
        "ts_pct": float(home["HOME_TS_PCT"]),
        "efg_pct": float(home["HOME_EFG_PCT"]),
        "pie": float(home["HOME_PIE"]),
        "last10": float(home["HOME_LAST10_WIN_PCT"]),
        "win_streak": int(home["HOME_WIN_STREAK"]),
        "rest_days": float(home["HOME_REST_DAYS"])

    }

    # ==========================================================
    # AWAY TEAM STATS
    # ==========================================================

    away_stats = {

        "elo_rating": float(away["AWAY_ELO"]),
        "net_rating": float(away["AWAY_NET_RATING"]),
        "off_rating": float(away["AWAY_OFF_RATING"]),
        "def_rating": float(away["AWAY_DEF_RATING"]),
        "pace": float(away["AWAY_PACE"]),
        "ts_pct": float(away["AWAY_TS_PCT"]),
        "efg_pct": float(away["AWAY_EFG_PCT"]),
        "pie": float(away["AWAY_PIE"]),
        "last10": float(away["AWAY_LAST10_WIN_PCT"]),
        "win_streak": int(away["AWAY_WIN_STREAK"]),
        "rest_days": float(away["AWAY_REST_DAYS"])

    }

    # ==========================================================
    # FEATURE VECTOR
    # ==========================================================

    feature_vector = {

        "NET_RATING_DIFF":
            home["HOME_NET_RATING"] - away["AWAY_NET_RATING"],

        "OFF_RATING_DIFF":
            home["HOME_OFF_RATING"] - away["AWAY_OFF_RATING"],

        "DEF_RATING_DIFF":
            home["HOME_DEF_RATING"] - away["AWAY_DEF_RATING"],

        "PACE_DIFF":
            home["HOME_PACE"] - away["AWAY_PACE"],

        "TS_PCT_DIFF":
            home["HOME_TS_PCT"] - away["AWAY_TS_PCT"],

        "EFG_PCT_DIFF":
            home["HOME_EFG_PCT"] - away["AWAY_EFG_PCT"],

        "PIE_DIFF":
            home["HOME_PIE"] - away["AWAY_PIE"],

        "HOME_LAST10_WIN_PCT":
            home["HOME_LAST10_WIN_PCT"],

        "AWAY_LAST10_WIN_PCT":
            away["AWAY_LAST10_WIN_PCT"],

        "LAST10_DIFF":
            home["HOME_LAST10_WIN_PCT"] -
            away["AWAY_LAST10_WIN_PCT"],

        "HOME_WIN_STREAK":
            home["HOME_WIN_STREAK"],

        "AWAY_WIN_STREAK":
            away["AWAY_WIN_STREAK"],

        "STREAK_DIFF":
            home["HOME_WIN_STREAK"] -
            away["AWAY_WIN_STREAK"],

        "HOME_REST_DAYS":
            home["HOME_REST_DAYS"],

        "AWAY_REST_DAYS":
            away["AWAY_REST_DAYS"],

        "REST_DAY_DIFF":
            home["HOME_REST_DAYS"] -
            away["AWAY_REST_DAYS"],

        "HOME_ELO":
            home["HOME_ELO"],

        "AWAY_ELO":
            away["AWAY_ELO"],

        "ELO_DIFF":
            home["HOME_ELO"] -
            away["AWAY_ELO"]

    }

    X = pd.DataFrame([feature_vector])

    return X, home_stats, away_stats


# ==========================================================
# PREDICT GAME
# ==========================================================

def predict_game(home_team, away_team, model_name):

    if home_team == away_team:

        raise ValueError(
            "Home Team and Away Team cannot be the same."
        )

    # ------------------------------------------------------

    model = get_model(model_name)

    X, home_stats, away_stats = build_feature_vector(
        home_team,
        away_team
    )

    # Logistic Regression needs scaling

    if model_name == "Logistic Regression":

        X_input = scaler.transform(X)

    else:

        X_input = X

    # ------------------------------------------------------

    probability = model.predict_proba(X_input)[0]

    home_probability = float(probability[1])
    away_probability = float(probability[0])

    # ------------------------------------------------------

    if home_probability >= away_probability:

        winner = home_team

    else:

        winner = away_team

    # ------------------------------------------------------

    confidence = max(
        home_probability,
        away_probability
    )

    if confidence < 0.60:

        confidence_level = "Toss-Up"

    elif confidence < 0.70:

        confidence_level = "Moderate"

    elif confidence < 0.80:

        confidence_level = "High"

    else:

        confidence_level = "Very High"

    # ------------------------------------------------------

    return {

        "prediction": {

            "winner": winner,

            "home_probability": round(
                home_probability * 100,
                2
            ),

            "away_probability": round(
                away_probability * 100,
                2
            ),

            "confidence": confidence_level

        },

        "selected_home_team": home_team,

        "selected_away_team": away_team,

        "home_stats": home_stats,

        "away_stats": away_stats,

        "feature_vector": X

    }

if __name__ == "__main__":

    result = predict_game(

        "Boston Celtics",
        "Denver Nuggets",
        "Logistic Regression"

    )

    print("=" * 60)

    print("Prediction")
    print(result["prediction"])

    print()

    print("Selected Home Team")
    print(result["selected_home_team"])

    print()

    print("Selected Away Team")
    print(result["selected_away_team"])

    print()

    print("Home Statistics")
    print(result["home_stats"])

    print()

    print("Away Statistics")
    print(result["away_stats"])

    print()

    print("Feature Vector")
    print(result["feature_vector"])

    print("=" * 60)
  