import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv(
    "data/processed/final_dataset_dynamic.csv"
)

# Better looking plots
sns.set_theme(style="whitegrid")

# ==========================================================
# DATASET OVERVIEW
# ==========================================================

def dataset_overview():

    print("=" * 70)
    print("NBA PLAYOFF PREDICTOR - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print(f"\nRows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nFirst 5 Rows")
    print(df.head())

    print("\nData Types")
    print(df.dtypes)

    print("\nSummary Statistics")
    print(df.describe(include="all"))

    # ==========================================================
# TARGET DISTRIBUTION
# ==========================================================

def target_distribution():

    plt.figure(figsize=(6,5))

    sns.countplot(
        data=df,
        x="HOME_WIN"
    )

    plt.title("Home Win Distribution")

    plt.xlabel("Home Win")

    plt.ylabel("Number of Games")

    plt.tight_layout()

    plt.savefig(
        "reports/figures/target_distribution.png",
        dpi=300
    )

    plt.show()

    print("\n Target Distribution Saved")


 # ==========================================================
# FEATURE HISTOGRAMS
# ==========================================================

def feature_histograms():

    features = [
        "ELO_DIFF",
        "NET_RATING_DIFF",
        "LAST10_DIFF",
        "STREAK_DIFF",
        "REST_DAY_DIFF",
        "TS_PCT_DIFF",
        "PIE_DIFF"
    ]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(16, 12)
    )

    axes = axes.flatten()

    for i, feature in enumerate(features):

        sns.histplot(
            data=df,
            x=feature,
            kde=True,
            ax=axes[i]
        )

        axes[i].set_title(feature)

    # Remove unused subplots
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    plt.savefig(
        "reports/figures/feature_histograms.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(" Feature Histograms Saved")


# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

def correlation_heatmap():

    features = [
        "HOME_WIN",
        "ELO_DIFF",
        "NET_RATING_DIFF",
        "OFF_RATING_DIFF",
        "DEF_RATING_DIFF",
        "LAST10_DIFF",
        "STREAK_DIFF",
        "REST_DAY_DIFF",
        "PACE_DIFF",
        "TS_PCT_DIFF",
        "EFG_PCT_DIFF",
        "PIE_DIFF"
    ]

    corr = df[features].corr()

    plt.figure(figsize=(12,10))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f"
    )

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        "reports/figures/correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(" Correlation Heatmap Saved")

# ==========================================================
# FEATURE VS TARGET CORRELATION
# ==========================================================

def feature_target_correlation():

    correlations = (
        df.corr(numeric_only=True)["HOME_WIN"]
        .sort_values(ascending=False)
    )

    correlations.to_csv(
        "reports/target_correlation.csv"
    )

    print("\n" + "=" * 60)
    print("FEATURE CORRELATION WITH HOME_WIN")
    print("=" * 60)

    print(correlations)

    print("\n Target Correlation Saved")    


# ==========================================================
# MAIN
# ==========================================================
def main():

    dataset_overview()

    target_distribution()

    feature_histograms()

    correlation_heatmap()

    feature_target_correlation()


if __name__ == "__main__":
    main()