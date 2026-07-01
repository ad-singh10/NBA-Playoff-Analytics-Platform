import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


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
# STRATIFIED K-FOLD
# ==========================================================

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)



# ==========================================================
# SCORING
# ==========================================================

scoring = [

    "accuracy",

    "precision",

    "recall",

    "f1",

    "roc_auc"

]


#Pipelines

logistic_pipeline = Pipeline(

    [

        ("scaler", StandardScaler()),

        (

            "model",

            LogisticRegression(

                random_state=42,

                max_iter=1000

            )

        )

    ]

)

random_forest_pipeline = Pipeline(

    [

        (

            "model",

            RandomForestClassifier(

                n_estimators=300,

                max_depth=10,

                random_state=42

            )

        )

    ]

)

xgboost_pipeline = Pipeline(

    [

        (

            "model",

            XGBClassifier(

                n_estimators=300,

                learning_rate=0.05,

                max_depth=4,

                subsample=0.8,

                colsample_bytree=0.8,

                random_state=42,

                eval_metric="logloss"

            )

        )

    ]

)


# ==========================================================
# CROSS VALIDATION FUNCTION
# ==========================================================

def validate_model(model, model_name):

    scores = cross_validate(

        estimator=model,

        X=X,

        y=y,

        cv=cv,

        scoring=scoring,

        return_train_score=False

    )

    results = {

        "Model": model_name,

        "Accuracy Mean": scores["test_accuracy"].mean(),

        "Accuracy Std": scores["test_accuracy"].std(),

        "Precision Mean": scores["test_precision"].mean(),

        "Recall Mean": scores["test_recall"].mean(),

        "F1 Mean": scores["test_f1"].mean(),

        "ROC AUC Mean": scores["test_roc_auc"].mean()

    }

    return results



# ==========================================================
# VALIDATE ALL MODELS
# ==========================================================

results = []

results.append(

    validate_model(

        logistic_pipeline,

        "Logistic Regression"

    )

)

results.append(

    validate_model(

        random_forest_pipeline,

        "Random Forest"

    )

)

results.append(

    validate_model(

        xgboost_pipeline,

        "XGBoost"

    )

)



# ==========================================================
# RESULTS DATAFRAME
# ==========================================================

results_df = pd.DataFrame(results)


print("\n")
print("="*70)
print("5-FOLD STRATIFIED CROSS VALIDATION")
print("="*70)

print(results_df)


results_df.to_csv(

    "reports/metrics/cross_validation_results.csv",

    index=False

)

print("\n✓ Cross Validation Results Saved")