
# Cell 1: Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set the project paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

print("Libraries imported successfully!")
print("Data folder:", DATA_DIR.resolve())
print("Output folder:", OUTPUT_DIR.resolve())




# Cell 2: Load all five datasets

files = {
    "claims": "claims.csv",
    "policies": "policies.csv",
    "customers": "customers.csv",
    "incidents": "incidents.csv",
    "processing": "processing.csv"
}

data = {}

for name, filename in files.items():
    path = DATA_DIR / filename

    if path.exists():
        data[name] = pd.read_csv(path)
        print(f"\n{name.upper()} DATASET")
        print("Shape:", data[name].shape)
        print("Columns:", data[name].columns.tolist())
        print(data[name].head())
    else:
        print(f"\nMissing file: {path}")



# Cell 3: Data quality report

for name, df in data.items():
    print(f"\n{'=' * 50}")
    print(f"DATA QUALITY REPORT: {name.upper()}")
    print(f"{'=' * 50}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nDescriptive statistics:")
    print(df.describe(include="all").T)






# ============================================
# STEP 2: DATA CLEANING
# ============================================

# Create working copies so original data stays unchanged
cleaned = {name: df.copy() for name, df in data.items()}

# 1. Remove exact duplicate rows from claims
before = len(cleaned["claims"])

cleaned["claims"] = cleaned["claims"].drop_duplicates()

print("Exact duplicate claim rows removed:",
      before - len(cleaned["claims"]))

# 2. Convert claim dates into datetime format
cleaned["claims"]["claim_date"] = pd.to_datetime(
    cleaned["claims"]["claim_date"],
    errors="coerce"
)

# 3. Standardize claim types
cleaned["claims"]["claim_type"] = (
    cleaned["claims"]["claim_type"]
    .astype("string")
    .str.strip()
    .str.title()
)

# 4. Handle missing claim amounts using the median
claim_median = cleaned["claims"]["claim_amount"].median()

cleaned["claims"]["claim_amount"] = (
    cleaned["claims"]["claim_amount"]
    .fillna(claim_median)
)

# 5. Handle missing incident details
cleaned["incidents"]["incident_type"] = (
    cleaned["incidents"]["incident_type"]
    .fillna("Unknown")
)

cleaned["incidents"]["severity"] = (
    cleaned["incidents"]["severity"]
    .fillna("Unknown")
)

# Standardize text columns
for column in ["incident_type", "location", "severity"]:
    cleaned["incidents"][column] = (
        cleaned["incidents"][column]
        .astype("string")
        .str.strip()
        .str.title()
    )

# 6. Identify extreme claim amounts using the IQR method
amounts = cleaned["claims"]["claim_amount"]

Q1 = amounts.quantile(0.25)
Q3 = amounts.quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

cleaned["claims"]["is_amount_outlier"] = (
    (amounts < lower_limit) |
    (amounts > upper_limit)
)

print("\nClaim amount outliers flagged:",
      cleaned["claims"]["is_amount_outlier"].sum())

print("Lower outlier limit:", round(lower_limit, 2))
print("Upper outlier limit:", round(upper_limit, 2))

# 7. Handle invalid settlement durations
processing_df = cleaned["processing"]

processing_df["invalid_settlement"] = (
    processing_df["settlement_days"].isna() |
    (processing_df["settlement_days"] < 0)
)

invalid_count = processing_df["invalid_settlement"].sum()

# Replace invalid durations with NaN first
processing_df.loc[
    processing_df["invalid_settlement"],
    "settlement_days"
] = np.nan

# Impute invalid durations with the median of valid durations
valid_median = processing_df["settlement_days"].median()

processing_df["settlement_days"] = (
    processing_df["settlement_days"]
    .fillna(valid_median)
)

print("\nInvalid settlement durations identified:",
      invalid_count)
print("Median valid settlement duration:", valid_median)

# 8. Save all cleaned datasets
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for name, df in cleaned.items():
    df.to_csv(
        OUTPUT_DIR / f"{name}_cleaned.csv",
        index=False
    )
    print(f"Saved: {name}_cleaned.csv")

# 9. Check cleaned data
print("\n========== CLEANING SUMMARY ==========")

for name, df in cleaned.items():
    print(f"\n{name.upper()}")
    print("Rows:", len(df))
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())

print("\nData cleaning completed!")




# ============================================
# STEP 3: MERGE CLEANED DATASETS
# ============================================

claims_clean = cleaned["claims"]
policies_clean = cleaned["policies"]
customers_clean = cleaned["customers"]
incidents_clean = cleaned["incidents"]
processing_clean = cleaned["processing"]

# Merge claims with policy information
merged = claims_clean.merge(
    policies_clean,
    on="policy_id",
    how="left",
    validate="many_to_one"
)

# Add customer information
merged = merged.merge(
    customers_clean,
    on="customer_id",
    how="left",
    validate="many_to_one"
)

# Add incident information
merged = merged.merge(
    incidents_clean,
    on="claim_number",
    how="left",
    validate="one_to_one"
)

# Add processing information
merged = merged.merge(
    processing_clean,
    on="claim_number",
    how="left",
    validate="one_to_one"
)

# Save the combined dataset
merged.to_csv(
    OUTPUT_DIR / "insurance_claims_combined.csv",
    index=False
)

print("Combined dataset shape:", merged.shape)
print("\nCombined columns:")
print(merged.columns.tolist())

print("\nMissing values after merging:")
print(merged.isnull().sum())

print("\nFirst five combined records:")
print(merged.head())

print("\nCombined dataset saved successfully!")




# ============================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================

# Use the merged dataset created in Step 3
eda = merged.copy()

# Ensure correct data types
eda["claim_date"] = pd.to_datetime(eda["claim_date"], errors="coerce")

# Create a folder for charts
CHART_DIR = OUTPUT_DIR / "eda_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

print("\nEDA DATASET SHAPE:", eda.shape)
print("\nNumeric summary:")
print(eda.describe().round(2))

# --------------------------------------------
# QUESTION 1: Which claim types have higher
# average claim value?
# --------------------------------------------

claim_type_analysis = (
    eda.groupby("claim_type")
    .agg(
        claim_count=("claim_number", "nunique"),
        average_claim_value=("claim_amount", "mean"),
        total_claim_value=("claim_amount", "sum")
    )
    .sort_values("average_claim_value", ascending=False)
)

print("\n1. CLAIM VALUE BY TYPE")
print(claim_type_analysis.round(2))

claim_type_analysis.to_csv(
    OUTPUT_DIR / "claim_value_by_type.csv"
)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=eda,
    x="claim_type",
    y="claim_amount",
    estimator=np.mean,
    errorbar=None
)
plt.title("Average Claim Amount by Claim Type")
plt.xlabel("Claim Type")
plt.ylabel("Average Claim Amount")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(CHART_DIR / "01_claim_value_by_type.png", dpi=300)
plt.show()


# --------------------------------------------
# QUESTION 2: Which regions generate more claims?
# --------------------------------------------

region_analysis = (
    eda.groupby("region")
    .agg(
        claim_count=("claim_number", "nunique"),
        total_claim_value=("claim_amount", "sum"),
        average_claim_value=("claim_amount", "mean")
    )
    .sort_values("claim_count", ascending=False)
)

print("\n2. REGIONAL CLAIM ANALYSIS")
print(region_analysis.round(2))

region_analysis.to_csv(
    OUTPUT_DIR / "regional_claim_analysis.csv"
)

plt.figure(figsize=(8, 5))
sns.countplot(
    data=eda,
    x="region",
    order=eda["region"].value_counts().index
)
plt.title("Number of Claims by Region")
plt.xlabel("Region")
plt.ylabel("Number of Claims")
plt.tight_layout()
plt.savefig(CHART_DIR / "02_claims_by_region.png", dpi=300)
plt.show()


# --------------------------------------------
# QUESTION 3: What is associated with longer
# settlement time?
# --------------------------------------------

settlement_by_type = (
    eda.groupby("claim_type")
    .agg(
        average_days=("settlement_days", "mean"),
        median_days=("settlement_days", "median"),
        claim_count=("claim_number", "nunique")
    )
    .sort_values("average_days", ascending=False)
)

settlement_by_severity = (
    eda.groupby("severity")
    .agg(
        average_days=("settlement_days", "mean"),
        claim_count=("claim_number", "nunique")
    )
    .sort_values("average_days", ascending=False)
)

settlement_by_status = (
    eda.groupby("claim_status")
    .agg(
        average_days=("settlement_days", "mean"),
        claim_count=("claim_number", "nunique")
    )
    .sort_values("average_days", ascending=False)
)

print("\n3A. SETTLEMENT TIME BY CLAIM TYPE")
print(settlement_by_type.round(2))

print("\n3B. SETTLEMENT TIME BY SEVERITY")
print(settlement_by_severity.round(2))

print("\n3C. SETTLEMENT TIME BY CLAIM STATUS")
print(settlement_by_status.round(2))

settlement_by_type.to_csv(
    OUTPUT_DIR / "settlement_by_type.csv"
)
settlement_by_severity.to_csv(
    OUTPUT_DIR / "settlement_by_severity.csv"
)
settlement_by_status.to_csv(
    OUTPUT_DIR / "settlement_by_status.csv"
)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=eda,
    x="claim_type",
    y="settlement_days",
    estimator=np.mean,
    errorbar=None
)
plt.title("Average Settlement Days by Claim Type")
plt.xlabel("Claim Type")
plt.ylabel("Average Settlement Days")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(CHART_DIR / "03_settlement_by_type.png", dpi=300)
plt.show()


# --------------------------------------------
# QUESTION 4: Which recorded process stages
# have longer settlement times?
# --------------------------------------------

stage_analysis = (
    eda.groupby("process_stage")
    .agg(
        claims=("claim_number", "nunique"),
        average_settlement_days=("settlement_days", "mean"),
        median_settlement_days=("settlement_days", "median"),
        total_claim_value=("claim_amount", "sum")
    )
    .sort_values("average_settlement_days", ascending=False)
)

print("\n4. PROCESS STAGE ANALYSIS")
print(stage_analysis.round(2))

stage_analysis.to_csv(
    OUTPUT_DIR / "process_stage_analysis.csv"
)

plt.figure(figsize=(10, 5))
sns.barplot(
    data=eda,
    x="process_stage",
    y="settlement_days",
    estimator=np.mean,
    errorbar=None
)
plt.title("Average Settlement Days by Recorded Process Stage")
plt.xlabel("Recorded Process Stage")
plt.ylabel("Average Settlement Days")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(CHART_DIR / "04_process_stage.png", dpi=300)
plt.show()


# --------------------------------------------
# QUESTION 5: How do claim values vary by severity?
# --------------------------------------------

severity_analysis = (
    eda.groupby("severity")
    .agg(
        claim_count=("claim_number", "nunique"),
        average_claim_value=("claim_amount", "mean"),
        median_claim_value=("claim_amount", "median"),
        total_claim_value=("claim_amount", "sum")
    )
    .sort_values("average_claim_value", ascending=False)
)

print("\n5. CLAIM VALUE BY SEVERITY")
print(severity_analysis.round(2))

severity_analysis.to_csv(
    OUTPUT_DIR / "claim_value_by_severity.csv"
)

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=eda,
    x="severity",
    y="claim_amount"
)
plt.title("Claim Amount Distribution by Severity")
plt.xlabel("Severity")
plt.ylabel("Claim Amount")
plt.tight_layout()
plt.savefig(CHART_DIR / "05_severity_comparison.png", dpi=300)
plt.show()


# --------------------------------------------
# QUESTION 6: Which categories need more
# investigation effort?
# --------------------------------------------

investigation_df = eda[
    eda["process_stage"].str.contains(
        "investigation", case=False, na=False
    )
].copy()

print("\n6A. INVESTIGATION WORKLOAD")
print("Claims recorded at Investigation stage:",
      investigation_df["claim_number"].nunique())

investigation_by_type = (
    investigation_df.groupby("claim_type")
    .agg(
        investigation_claims=("claim_number", "nunique"),
        average_claim_value=("claim_amount", "mean"),
        average_settlement_days=("settlement_days", "mean")
    )
    .sort_values("investigation_claims", ascending=False)
)

print("\nInvestigation workload by claim type:")
print(investigation_by_type.round(2))

investigation_by_type.to_csv(
    OUTPUT_DIR / "investigation_by_claim_type.csv"
)

# Workload by assigned agent across all recorded claims
agent_workload = (
    eda.groupby("assignee")
    .agg(
        assigned_claims=("claim_number", "nunique"),
        average_settlement_days=("settlement_days", "mean"),
        total_claim_value=("claim_amount", "sum")
    )
    .sort_values("assigned_claims", ascending=False)
)

print("\nWorkload by assigned agent:")
print(agent_workload.round(2))

agent_workload.to_csv(
    OUTPUT_DIR / "agent_workload.csv"
)

plt.figure(figsize=(9, 5))
sns.countplot(
    data=eda,
    y="assignee",
    order=eda["assignee"].value_counts().index
)
plt.title("Number of Claims Assigned to Each Agent")
plt.xlabel("Number of Claims")
plt.ylabel("Assignee")
plt.tight_layout()
plt.savefig(CHART_DIR / "06_agent_workload.png", dpi=300)
plt.show()


# --------------------------------------------
# BONUS: Correlation analysis
# --------------------------------------------

numeric_cols = [
    "claim_amount",
    "premium",
    "tenure_years",
    "customer_tenure_years",
    "settlement_days"
]

correlation = eda[numeric_cols].corr()

print("\nNUMERIC CORRELATION MATRIX")
print(correlation.round(2))

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Between Numeric Variables")
plt.tight_layout()
plt.savefig(CHART_DIR / "07_correlation_heatmap.png", dpi=300)
plt.show()


# --------------------------------------------
# Save final EDA dataset
# --------------------------------------------

eda.to_csv(
    OUTPUT_DIR / "eda_analysis_dataset.csv",
    index=False
)

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Analysis tables saved in:", OUTPUT_DIR.resolve())
print("Charts saved in:", CHART_DIR.resolve())




