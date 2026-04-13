from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import zscore
import numpy as np
from sklearn.ensemble import IsolationForest


DATA_PATH = Path("data/processed_transactions.csv")
output_path = Path("phase1_static/outlier_detection")
output_path.mkdir(parents=True, exist_ok=True)
NUMERIC_COLUMNS = ["Quantity", "Price", "LineTotal"]

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing file: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("\nNull counts:")
print(df[NUMERIC_COLUMNS].isna().sum())

print("\nBasic summary:")
print(tabulate(df[NUMERIC_COLUMNS].describe(), headers='keys', tablefmt='psql', floatfmt=",.2f"))


# -------------------------
# IQR
# -------------------------
print("\nIQR:")
mask = pd.Series(True, index=df.index)

for col in NUMERIC_COLUMNS:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    mask &= df[col].between(lower, upper)

df_clean = df[mask]

print("Original shape:", df.shape)
print("Cleaned shape:", df_clean.shape)
print("Percentage Removed (%):", (1 - len(df_clean)/len(df)) * 100)

sns.set_style("whitegrid")

for col in NUMERIC_COLUMNS:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(y=df[col], ax=axes[0])
    axes[0].set_title(f"{col} BEFORE")

    sns.boxplot(y=df_clean[col], ax=axes[1])
    axes[1].set_title(f"{col} AFTER (IQR)")

    plt.tight_layout()

    # Save instead of just showing
    filename = output_path / f"{col.lower()}_iqr_before_after.png"
    plt.savefig(filename, dpi=300)

    plt.close()

df_clean.to_csv(output_path / "cleaned_iqr.csv", index=False)
print("Saved cleaned_iqr.csv")

# -------------------------
# Z-Score
# -------------------------
print("\nZ-Score:")
z_scores = np.abs(zscore(df[NUMERIC_COLUMNS]))

if len(NUMERIC_COLUMNS) == 1:
    mask = z_scores < 3
else:
    mask = (z_scores < 3).all(axis=1)

df_z = df[mask]

print("Original shape:", df.shape)
print("Z-score shape:", df_z.shape)
print("Removed %:", (1 - len(df_z)/len(df)) * 100)

sns.set_style("whitegrid")

for col in NUMERIC_COLUMNS:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(y=df[col], ax=axes[0])
    axes[0].set_title(f"{col} BEFORE")

    sns.boxplot(y=df_z[col], ax=axes[1])
    axes[1].set_title(f"{col} AFTER (Z-score)")

    plt.tight_layout()

    filename = output_path / f"{col.lower()}_zscore_before_after.png"
    plt.savefig(filename, dpi=300)

    plt.close()

# -------------------------
# Isolation Forest
# -------------------------
print("\nIsolation Forest:")
iso = IsolationForest(
    contamination=0.02,
    random_state=42,
    n_estimators=200
)

preds = iso.fit_predict(df[NUMERIC_COLUMNS])
df_iso = df[preds == 1]

print("Original shape:", df.shape)
print("Isolation Forest shape:", df_iso.shape)
print("Removed %:", (1 - len(df_iso)/len(df)) * 100)

for col in NUMERIC_COLUMNS:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(y=df[col], ax=axes[0])
    axes[0].set_title(f"{col} BEFORE")

    sns.boxplot(y=df_iso[col], ax=axes[1])
    axes[1].set_title(f"{col} AFTER (Isolation Forest)")

    plt.tight_layout()

    filename = output_path / f"{col.lower()}_isolation_before_after.png"
    plt.savefig(filename, dpi=300)

    plt.close()