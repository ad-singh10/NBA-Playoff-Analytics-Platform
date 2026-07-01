import os
import numpy as np
import pandas as pd

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("data/processed/final_dataset_dynamic.csv")

# ==========================================================
# CREATE REPORTS FOLDER
# ==========================================================

os.makedirs("reports", exist_ok=True)

report_path = "reports/dataset_report.txt"

report = []

# ==========================================================
# DATASET INFORMATION
# ==========================================================

report.append("=" * 60)
report.append("NBA PLAYOFF PREDICTOR - DATASET VALIDATION REPORT")
report.append("=" * 60)

report.append(f"\nRows    : {df.shape[0]}")
report.append(f"Columns : {df.shape[1]}")

# ==========================================================
# DATA TYPES
# ==========================================================

report.append("\n" + "=" * 60)
report.append("DATA TYPES")
report.append("=" * 60)

report.append(df.dtypes.to_string())

# ==========================================================
# MISSING VALUES
# ==========================================================

report.append("\n" + "=" * 60)
report.append("MISSING VALUES")
report.append("=" * 60)

missing = df.isnull().sum()

report.append(missing.to_string())

# ==========================================================
# DUPLICATES
# ==========================================================

report.append("\n" + "=" * 60)
report.append("DUPLICATES")
report.append("=" * 60)

report.append(f"Duplicate Rows : {df.duplicated().sum()}")

if "GAME_ID" in df.columns:
    report.append(
        f"Duplicate GAME_ID : {df['GAME_ID'].duplicated().sum()}"
    )

# ==========================================================
# INFINITE VALUES
# ==========================================================

numeric_df = df.select_dtypes(include=np.number)

infinite_values = np.isinf(numeric_df).sum().sum()

report.append("\n" + "=" * 60)
report.append("INFINITE VALUES")
report.append("=" * 60)

report.append(f"Infinite Values : {infinite_values}")

# ==========================================================
# CONSTANT FEATURES
# ==========================================================

constant_cols = [
    col for col in numeric_df.columns
    if numeric_df[col].nunique() == 1
]

report.append("\n" + "=" * 60)
report.append("CONSTANT FEATURES")
report.append("=" * 60)

if constant_cols:
    report.append(str(constant_cols))
else:
    report.append("None")

# ==========================================================
# TARGET DISTRIBUTION
# ==========================================================

report.append("\n" + "=" * 60)
report.append("TARGET DISTRIBUTION")
report.append("=" * 60)

target_counts = df["HOME_WIN"].value_counts()

target_percent = (
    df["HOME_WIN"]
    .value_counts(normalize=True)
    * 100
)

for cls in target_counts.index:
    report.append(
        f"{cls} : {target_counts[cls]} ({target_percent[cls]:.2f}%)"
    )

# ==========================================================
# SUMMARY STATISTICS
# ==========================================================

report.append("\n" + "=" * 60)
report.append("SUMMARY STATISTICS")
report.append("=" * 60)

report.append(numeric_df.describe().to_string())

# ==========================================================
# SAVE REPORT
# ==========================================================

with open(report_path, "w") as f:
    f.write("\n".join(report))

# ==========================================================
# TERMINAL OUTPUT
# ==========================================================

print("=" * 60)
print("DATASET VALIDATION")
print("=" * 60)

print(f"Rows                : {df.shape[0]}")
print(f"Columns             : {df.shape[1]}")
print(f"Missing Values      : {missing.sum()}")
print(f"Duplicate Rows      : {df.duplicated().sum()}")
print(f"Infinite Values     : {infinite_values}")
print(f"Constant Features   : {len(constant_cols)}")

print("\nTarget Distribution")

print(target_percent)

print("\n✓ Dataset Validation Complete")
print(f"✓ Report Saved : {report_path}")