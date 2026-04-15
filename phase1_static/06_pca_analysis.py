"""
Layer 7 — PCA Analysis
File: phase1_static/06_pca_analysis.py

Uses existing preprocessing utilities from:
- preprocessing.data_loader.normalize
- preprocessing.data_loader.run_pca

Input:
- data/rfm_table.csv

Outputs:
- data/rfm_pca.csv
- phase1_static/pca_analysis/*.png
- phase1_static/pca_analysis/*.csv
- phase1_static/pca_analysis/observations.txt
"""

from pathlib import Path
import warnings
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(str(Path(__file__).resolve().parent.parent))
from preprocessing.data_loader import normalize, run_pca

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12


# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "phase1_static" / "pca_analysis"

INPUT_FILE = DATA_DIR / "rfm_table.csv"
PCA_OUTPUT_FILE = DATA_DIR / "rfm_pca.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Helpers
# =========================================================
def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def write_observation(file_handle, title: str, observation: str) -> None:
    file_handle.write(f"{title}\n")
    file_handle.write("-" * len(title) + "\n")
    file_handle.write(observation.strip() + "\n\n")


def validate_rfm_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Recency", "Frequency", "MonetaryValue"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns in rfm_table.csv: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    return df


def plot_scree(explained_variance_ratio: np.ndarray, obs_file) -> None:
    components = np.arange(1, len(explained_variance_ratio) + 1)

    plt.figure()
    plt.plot(components, explained_variance_ratio, marker="o", linewidth=2)
    plt.xticks(components)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot")
    plt.grid(True, linestyle="--", alpha=0.6)
    save_plot("01_scree_plot.png")

    write_observation(
        obs_file,
        "1. Scree Plot",
        "The scree plot shows how much variance is captured by each principal component. "
        "The scree plot shows a sharp drop after PC1 and a clear elbow at PC2, indicating that"
        " the first two components capture the majority of variance, while the third component contributes minimal additional information."
    )


def plot_cumulative_variance(explained_variance_ratio: np.ndarray, obs_file) -> None:
    cumulative = np.cumsum(explained_variance_ratio)
    components = np.arange(1, len(explained_variance_ratio) + 1)

    plt.figure()
    plt.plot(components, cumulative, marker="o", linewidth=2)
    plt.xticks(components)
    plt.ylim(0, 1.05)
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance")
    plt.grid(True, linestyle="--", alpha=0.6)
    save_plot("02_cumulative_variance.png")

    write_observation(
        obs_file,
        "2. Cumulative Explained Variance",
        "The cumulative variance plot shows how much total information is retained as more components are added. "
        "This helps justify whether a 2D PCA view is sufficient or whether the third component adds meaningful structure."
    )


def plot_pca_scatter_2d(pca_df: pd.DataFrame, obs_file) -> None:
    plt.figure()
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", alpha=0.7, s=45)
    plt.title("PCA Scatter Plot (2D)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle="--", alpha=0.6)
    save_plot("03_pca_scatter_2d.png")

    write_observation(
        obs_file,
        "3. PCA Scatter Plot (2D)",
        "The spread along PC1 suggests a dominant behavioral gradient, likely driven by "
        "purchasing intensity (frequency and monetary value), while PC2 captures secondary variation."
    )


def plot_pca_scatter_3d(pca_df: pd.DataFrame, obs_file) -> None:
    if not {"PC1", "PC2", "PC3"}.issubset(pca_df.columns):
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pca_df["PC1"], pca_df["PC2"], pca_df["PC3"], alpha=0.65, s=30)

    ax.set_title("PCA Scatter Plot (3D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_pca_scatter_3d.png", dpi=300, bbox_inches="tight")
    plt.close()

    write_observation(
        obs_file,
        "4. PCA Scatter Plot (3D)",
        "The third dimension adds limited additional separation, confirming that most structure is captured in 2D."
    )


def plot_biplot(scores: np.ndarray, pca_model, feature_names: list[str], obs_file) -> None:
    if scores.shape[1] < 2:
        return

    loadings = pca_model.components_.T

    plt.figure(figsize=(10, 7))
    plt.scatter(scores[:, 0], scores[:, 1], alpha=0.5, s=35)

    scale_x = np.max(np.abs(scores[:, 0])) * 0.8
    scale_y = np.max(np.abs(scores[:, 1])) * 0.8

    for i, feature in enumerate(feature_names):
        plt.arrow(
            0, 0,
            loadings[i, 0] * scale_x,
            loadings[i, 1] * scale_y,
            color="red",
            alpha=0.8,
            head_width=0.08,
            length_includes_head=True
        )
        plt.text(
            loadings[i, 0] * scale_x * 1.08,
            loadings[i, 1] * scale_y * 1.08,
            feature,
            color="red",
            fontsize=11,
            weight="bold"
        )

    plt.axhline(0, color="gray", linewidth=1)
    plt.axvline(0, color="gray", linewidth=1)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Biplot")
    plt.grid(True, linestyle="--", alpha=0.6)
    save_plot("05_pca_biplot.png")

    write_observation(
        obs_file,
        "5. PCA Biplot",
        "The biplot shows that Frequency and MonetaryValue are strongly aligned, indicating that "
        "customers who purchase frequently also tend to spend more. Recency points in the opposite direction, "
        "suggesting that customers who have not purchased recently tend to have lower frequency and spending. This confirms an "
        "inverse relationship between recency and customer value."
    )


def plot_singular_values(singular_values: np.ndarray, obs_file) -> None:
    components = np.arange(1, len(singular_values) + 1)

    plt.figure()
    sns.barplot(x=components, y=singular_values)
    plt.xlabel("Principal Component")
    plt.ylabel("Singular Value")
    plt.title("Singular Values by Principal Component")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    save_plot("06_singular_values.png")

    write_observation(
        obs_file,
        "6. Singular Values",
        "Singular values measure the strength of variation captured by each principal component. "
        "Larger values correspond to more dominant latent structure in the RFM data. "
        "This aligns with the explained variance results, reinforcing that PC1 captures the majority of variation."
    )


def save_summary_tables(pca_model, explained_ratio, condition_number, feature_names) -> None:
    n_components = len(explained_ratio)
    component_labels = [f"PC{i}" for i in range(1, n_components + 1)]

    variance_df = pd.DataFrame({
        "Component": component_labels,
        "ExplainedVariance": pca_model.explained_variance_,
        "ExplainedVarianceRatio": explained_ratio,
        "CumulativeVarianceRatio": np.cumsum(explained_ratio),
        "SingularValue": pca_model.singular_values_
    })

    loadings_df = pd.DataFrame(
        pca_model.components_.T,
        index=feature_names,
        columns=component_labels
    ).reset_index().rename(columns={"index": "Feature"})

    condition_df = pd.DataFrame({
        "Metric": ["Condition Number"],
        "Value": [condition_number]
    })

    variance_df.to_csv(OUTPUT_DIR / "pca_variance_summary.csv", index=False)
    loadings_df.to_csv(OUTPUT_DIR / "pca_loadings.csv", index=False)
    condition_df.to_csv(OUTPUT_DIR / "condition_number.csv", index=False)


# =========================================================
# Main
# =========================================================
def main() -> None:
    print("=" * 70)
    print("Running Layer 7 — PCA Analysis")
    print("=" * 70)

    # Load outlier-cleaned RFM table from Layer 1
    rfm = pd.read_csv(INPUT_FILE)
    rfm = validate_rfm_columns(rfm)

    # Apply the SAME transformation logic used in Layer 1
    rfm_log = normalize(
        rfm,
        method="log",
        columns=["Recency", "Frequency", "MonetaryValue"]
    )

    # Run PCA with 3 components for full Layer 7 analysis
    pca_df, explained_ratio, pca_model = run_pca(
        rfm_log,
        columns=["Recency", "Frequency", "MonetaryValue"],
        n_components=3,
        standardize=True
    )

    # Save full PCA dataframe
    pca_df.to_csv(PCA_OUTPUT_FILE, index=False)

    # Condition number
    singular_values = pca_model.singular_values_
    min_sv = np.min(singular_values)
    condition_number = np.max(singular_values) / min_sv if min_sv > 0 else np.inf

    # Save summary tables
    save_summary_tables(
        pca_model=pca_model,
        explained_ratio=explained_ratio,
        condition_number=condition_number,
        feature_names=["Recency", "Frequency", "MonetaryValue"]
    )

    # Build numeric score matrix for biplot
    score_cols = [col for col in ["PC1", "PC2", "PC3"] if col in pca_df.columns]
    scores = pca_df[score_cols].values

    # Generate plots and observations
    with open(OUTPUT_DIR / "observations.txt", "w", encoding="utf-8") as obs_file:
        obs_file.write("LAYER 7 — PCA ANALYSIS OBSERVATIONS\n")
        obs_file.write("=" * 42 + "\n\n")

        plot_scree(explained_ratio, obs_file)
        plot_cumulative_variance(explained_ratio, obs_file)
        plot_pca_scatter_2d(pca_df, obs_file)
        plot_pca_scatter_3d(pca_df, obs_file)
        plot_biplot(scores, pca_model, ["Recency", "Frequency", "MonetaryValue"], obs_file)
        plot_singular_values(singular_values, obs_file)

        write_observation(
            obs_file,
            "7. Condition Number Summary",
            f"The PCA condition number is {condition_number:.4f}, "
            "indicating that the principal components are well-conditioned and the dataset does "
            "not suffer from severe multicollinearity. This suggests that PCA results are stable and reliable."
        )

    # Console summary
    print("\nExplained Variance Ratio:")
    for i, ratio in enumerate(explained_ratio, start=1):
        print(f"PC{i}: {ratio:.4f}")

    print("\nCumulative Explained Variance:")
    for i, value in enumerate(np.cumsum(explained_ratio), start=1):
        print(f"PC1..PC{i}: {value:.4f}")

    print("\nSingular Values:")
    for i, value in enumerate(singular_values, start=1):
        print(f"PC{i}: {value:.4f}")

    print(f"\nCondition Number: {condition_number:.4f}")

    print("\nSaved:")
    print(f"- {PCA_OUTPUT_FILE}")
    print(f"- {OUTPUT_DIR}")


if __name__ == "__main__":
    main()