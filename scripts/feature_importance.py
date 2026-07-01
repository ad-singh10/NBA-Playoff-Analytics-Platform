import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# FEATURE NAMES
# ==========================================================

features = [

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

rf = joblib.load(
    "models/random_forest_tuned.pkl"
)

xgb = joblib.load(
    "models/xgboost_tuned.pkl"
)

# ==========================================================
# RANDOM FOREST IMPORTANCE
# ==========================================================

rf_importance = pd.DataFrame({

    "Feature": features,

    "Importance": rf.named_steps[
        "model"
    ].feature_importances_

})

rf_importance = rf_importance.sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10,8))

plt.barh(

    rf_importance["Feature"],

    rf_importance["Importance"]

)

plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(

    "reports/figures/random_forest_feature_importance.png",

    dpi=300

)

plt.show()

# ==========================================================
# XGBOOST IMPORTANCE
# ==========================================================

xgb_importance = pd.DataFrame({

    "Feature": features,

    "Importance": xgb.named_steps[
        "model"
    ].feature_importances_

})

xgb_importance = xgb_importance.sort_values(

    by="Importance",

    ascending=True

)

plt.figure(figsize=(10,8))

plt.barh(

    xgb_importance["Feature"],

    xgb_importance["Importance"]

)

plt.title("XGBoost Feature Importance")

plt.tight_layout()

plt.savefig(

    "reports/figures/xgboost_feature_importance.png",

    dpi=300

)

plt.show()

print("✓ Feature Importance Figures Saved")