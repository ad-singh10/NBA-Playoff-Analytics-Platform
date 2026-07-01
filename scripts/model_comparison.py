import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

results = pd.DataFrame({

    "Model":[

        "Logistic Regression",
        "Random Forest",
        "XGBoost"

    ],

    "Accuracy":[

        64.87,
        63.67,
        63.67

    ],

    "Precision":[

        66.91,
        67.15,
        67.89

    ],

    "Recall":[

        80.10,
        75.44,
        73.01

    ],

    "F1":[

        72.89,
        71.01,
        70.33

    ],

    "ROC AUC":[

        67.37,
        66.91,
        66.27

    ]

})

# ==========================================================
# BAR CHART
# ==========================================================

metrics = [

    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC AUC"

]

for metric in metrics:

    plt.figure(figsize=(8,5))

    plt.bar(

        results["Model"],

        results[metric]

    )

    plt.ylabel(metric)

    plt.title(f"{metric} Comparison")

    plt.tight_layout()

    plt.savefig(

        f"reports/figures/{metric.lower().replace(' ','_')}_comparison.png",

        dpi=300

    )

    plt.show()

print(" Model Comparison Charts Saved")