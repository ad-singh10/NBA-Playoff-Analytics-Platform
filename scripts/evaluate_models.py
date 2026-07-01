import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import train_test_split


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



X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)




# ==========================================================
# MODEL EVALUATION FUNCTION
# ==========================================================

def evaluate_model(model, X_test, y_test, model_name):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    print("\nClassification Report\n")

    print(classification_report(
        y_test,
        y_pred
    ))

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title(f"{model_name} Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        f"reports/figures/{model_name.lower().replace(' ','_')}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("✓ Confusion Matrix Saved")



# ==========================================================
# LOAD MODELS
# ==========================================================

logistic_model = joblib.load(
    "models/logistic_regression.pkl"
)

random_forest_model = joblib.load(
    "models/random_forest.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

# ==========================================================
# EVALUATE MODELS
# ==========================================================


X_test_scaled = scaler.transform(X_test)

evaluate_model(
    logistic_model,
    X_test_scaled,
    y_test,
    "Logistic Regression"
)

evaluate_model(
    random_forest_model,
    X_test,
    y_test,
    "Random Forest"
)

# ==========================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ==========================================================

feature_importance = pd.DataFrame({

    "Feature": features,

    "Importance": random_forest_model.feature_importances_

})

feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False

)

print("\n")
print("=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

print(feature_importance)

feature_importance.to_csv(

    "reports/feature_importance.csv",

    index=False

)

xgboost_model = joblib.load(
    "models/xgboost.pkl"
)


evaluate_model(
    xgboost_model,
    X_test,
    y_test,
    "XGBoost"
)

# ==========================================================
# XGBOOST FEATURE IMPORTANCE
# ==========================================================

feature_importance = pd.DataFrame({

    "Feature": features,

    "Importance": xgboost_model.feature_importances_

})

feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False

)

print("\n")
print("=" * 60)
print("XGBOOST FEATURE IMPORTANCE")
print("=" * 60)

print(feature_importance)

feature_importance.to_csv(

    "reports/xgboost_feature_importance.csv",

    index=False

)