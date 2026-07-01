import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv(
    "data/processed/final_dataset_dynamic.csv"
)

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
# SCALE DATA
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
# ROC CURVE
# ==========================================================

plt.figure(figsize=(8,6))

models = [

    ("Logistic Regression", logistic, X_test_scaled),

    ("Random Forest", rf, X_test),

    ("XGBoost", xgb, X_test)

]

for name, model, data in models:

    probabilities = model.predict_proba(data)[:,1]

    fpr, tpr, _ = roc_curve(

        y_test,

        probabilities

    )

    roc_auc = auc(

        fpr,

        tpr

    )

    plt.plot(

        fpr,

        tpr,

        linewidth=2,

        label=f"{name} (AUC={roc_auc:.3f})"

    )

plt.plot(

    [0,1],

    [0,1],

    "--"

)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(

    "reports/figures/roc_curve.png",

    dpi=300

)

plt.show()

print("✓ ROC Curve Saved")