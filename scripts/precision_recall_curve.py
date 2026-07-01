import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv(
    "data/processed/final_dataset_dynamic.csv"
)

# ==========================================================
# FEATURES
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

X = df[features]

y = df["HOME_WIN"]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    stratify=y,

    random_state=42

)

# ==========================================================
# LOAD SCALER
# ==========================================================

scaler = joblib.load(
    "models/scaler.pkl"
)

X_test_scaled = scaler.transform(X_test)

# ==========================================================
# LOAD MODELS
# ==========================================================

logistic = joblib.load(
    "models/logistic_regression.pkl"
)

rf = joblib.load(
    "models/random_forest_tuned.pkl"
)

xgb = joblib.load(
    "models/xgboost_tuned.pkl"
)

# ==========================================================
# PRECISION RECALL CURVE
# ==========================================================

plt.figure(figsize=(8,6))

models = [

    ("Logistic Regression", logistic, X_test_scaled),

    ("Random Forest", rf, X_test),

    ("XGBoost", xgb, X_test)

]

for name, model, data in models:

    probabilities = model.predict_proba(data)[:,1]

    precision, recall, _ = precision_recall_curve(

        y_test,

        probabilities

    )

    ap = average_precision_score(

        y_test,

        probabilities

    )

    plt.plot(

        recall,

        precision,

        linewidth=2,

        label=f"{name} (AP={ap:.3f})"

    )

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(

    "reports/figures/precision_recall_curve.png",

    dpi=300

)

plt.show()

print(" Precision-Recall Curve Saved")