import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, pearsonr, skew, spearmanr
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
DATA_PATH = "data/rfm_table.csv"
OUTPUT_DIR = "analysis/statistics_outputs"

PLOT_STYLE = "whitegrid"
RANDOM_STATE = 42
SCATTER_SAMPLE_SIZE = 1000
KDE_SAMPLE_SIZE = 1200
FIG_DPI = 300


# ============================================================
# Helpers
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_plot_theme() -> None:
    sns.set_theme(style=PLOT_STYLE, context="talk")
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "regular"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.35


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def safe_sample(df: pd.DataFrame, n: int, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=random_state)


def save_table(df: pd.DataFrame, filename: str, index: bool = True) -> None:
    out_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(out_path, index=index)


def save_text(lines: list[str], filename: str) -> None:
    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["Recency", "Frequency", "MonetaryValue"]
    return [col for col in preferred if col in df.columns]


# ============================================================
# Statistics
# ============================================================
def compute_statistics(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    stats_df = df[cols].describe().T
    stats_df["median"] = df[cols].median()
    stats_df["skewness"] = df[cols].apply(lambda x: skew(x.dropna()))
    stats_df["kurtosis"] = df[cols].apply(lambda x: kurtosis(x.dropna()))

    stats_df = stats_df[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "median", "skewness", "kurtosis"]
    ]

    return stats_df.round(4)


def correlation_with_pvalues(
    df: pd.DataFrame,
    cols: list[str],
    method: str = "pearson"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = df[cols].copy()

    for col in cols:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")

    corr = subset.corr(method=method)
    pvals = pd.DataFrame(np.nan, index=cols, columns=cols)

    for c1 in cols:
        for c2 in cols:
            if c1 == c2:
                pvals.loc[c1, c2] = 0.0
                continue

            valid = subset[[c1, c2]].dropna()

            if len(valid) < 3:
                continue

            x = valid.iloc[:, 0].to_numpy()
            y = valid.iloc[:, 1].to_numpy()

            if method == "pearson":
                _, p = pearsonr(x, y)
            else:
                _, p = spearmanr(x, y)

            pvals.loc[c1, c2] = p

    return corr.round(4), pvals.round(6)


# ============================================================
# Plotting
# ============================================================
def plot_heatmap(corr_df: pd.DataFrame, title: str, filename: str) -> None:
    plt.figure(figsize=(9, 7))
    ax = sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1.5,
        linecolor="white",
        cbar_kws={"shrink": 0.9, "label": "Correlation Coefficient"},
        annot_kws={"size": 16, "weight": "bold"}
    )

    ax.set_title(title, fontsize=22, pad=14, weight="bold")
    ax.set_xlabel("Variables", fontsize=15)
    ax.set_ylabel("Variables", fontsize=15)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def plot_scatter_matrix(df: pd.DataFrame, cols: list[str], filename: str) -> None:
    pair_df = df[cols].dropna().copy()
    pair_df = safe_sample(pair_df, SCATTER_SAMPLE_SIZE)

    g = sns.pairplot(
        pair_df,
        corner=True,
        diag_kind="kde",
        height=3.2,
        plot_kws={
            "s": 55,
            "alpha": 0.45,
            "edgecolor": "white",
            "linewidth": 0.4
        },
        diag_kws={
            "fill": True,
            "alpha": 0.65,
            "linewidth": 2
        }
    )

    g.fig.subplots_adjust(top=0.94)
    g.fig.suptitle("Scatter Matrix of RFM Variables", fontsize=24, weight="bold")

    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def plot_multivariate_kde(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    filename: str
) -> None:
    subset = df[[x_col, y_col]].dropna().copy()
    subset = safe_sample(subset, KDE_SAMPLE_SIZE)

    plt.figure(figsize=(10, 7))

    ax = sns.kdeplot(
        data=subset,
        x=x_col,
        y=y_col,
        fill=True,
        levels=8,
        thresh=0.08,
        bw_adjust=1.1,
        cmap="Blues",
        alpha=0.85
    )

    sns.scatterplot(
        data=subset,
        x=x_col,
        y=y_col,
        color="black",
        alpha=0.18,
        s=35,
        edgecolor=None
    )

    ax.set_title(
        f"Multivariate KDE Density: {x_col} vs {y_col}",
        fontsize=22,
        pad=14,
        weight="bold"
    )
    ax.set_xlabel(x_col, fontsize=15)
    ax.set_ylabel(y_col, fontsize=15)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


# ============================================================
# Observations
# ============================================================
def build_observations(
    stats_df: pd.DataFrame,
    pearson_corr: pd.DataFrame,
    spearman_corr: pd.DataFrame
) -> list[str]:
    lines = []

    lines.append("Layer 9 — Statistics & Correlation Observations")
    lines.append("=" * 60)
    lines.append("")

    lines.append("1. Descriptive Statistics")
    lines.append("-" * 30)
    lines.append(
        f"Recency has a mean of {stats_df.loc['Recency', 'mean']:.2f} and a median of "
        f"{stats_df.loc['Recency', 'median']:.2f}, indicating a right-skewed distribution with many relatively recent customers and a long tail of inactive customers."
    )
    lines.append(
        f"Frequency has a mean of {stats_df.loc['Frequency', 'mean']:.2f}, a median of "
        f"{stats_df.loc['Frequency', 'median']:.2f}, and a skewness of {stats_df.loc['Frequency', 'skewness']:.2f}, showing that most customers purchase infrequently while a smaller group purchases much more often."
    )
    lines.append(
        f"MonetaryValue has a mean of {stats_df.loc['MonetaryValue', 'mean']:.2f}, a median of "
        f"{stats_df.loc['MonetaryValue', 'median']:.2f}, and a skewness of {stats_df.loc['MonetaryValue', 'skewness']:.2f}, suggesting that a relatively small number of customers account for disproportionately high spending."
    )
    lines.append("")

    lines.append("2. Pearson Correlation")
    lines.append("-" * 30)
    lines.append(
        f"Frequency and MonetaryValue have a strong positive Pearson correlation of {pearson_corr.loc['Frequency', 'MonetaryValue']:.2f}, indicating that customers who buy more frequently also tend to spend more overall."
    )
    lines.append(
        f"Recency and Frequency have a moderate negative Pearson correlation of {pearson_corr.loc['Recency', 'Frequency']:.2f}, suggesting that customers who purchased more recently tend to purchase more often."
    )
    lines.append(
        f"Recency and MonetaryValue have a moderate negative Pearson correlation of {pearson_corr.loc['Recency', 'MonetaryValue']:.2f}, showing that customers with higher inactivity generally contribute lower spending."
    )
    lines.append("")

    lines.append("3. Spearman Correlation")
    lines.append("-" * 30)
    lines.append(
        f"Frequency and MonetaryValue have a strong positive Spearman correlation of {spearman_corr.loc['Frequency', 'MonetaryValue']:.2f}, confirming a strong monotonic relationship between repeat purchasing and total spending."
    )
    lines.append(
        f"Recency and Frequency have a Spearman correlation of {spearman_corr.loc['Recency', 'Frequency']:.2f}, reinforcing the inverse relationship between inactivity and purchase frequency."
    )
    lines.append(
        f"Recency and MonetaryValue have a Spearman correlation of {spearman_corr.loc['Recency', 'MonetaryValue']:.2f}, which supports the pattern that recently active customers tend to have greater monetary value."
    )
    lines.append(
        "The Spearman coefficients are slightly stronger than the Pearson coefficients, suggesting that the relationships are monotonic and not perfectly linear."
    )
    lines.append("")

    lines.append("4. Scatter Matrix")
    lines.append("-" * 30)
    lines.append(
        "The scatter matrix shows strong clustering in the lower ranges of Frequency and MonetaryValue, which is typical of retail data where most customers are low-to-moderate buyers."
    )
    lines.append(
        "The vertical banding in Frequency is expected because Frequency is a discrete count variable rather than a continuous measurement."
    )
    lines.append(
        "The negative spread between Recency and the other variables suggests that more recently active customers are generally more frequent buyers and stronger contributors to revenue."
    )
    lines.append("")

    lines.append("5. Multivariate KDE")
    lines.append("-" * 30)
    lines.append(
        "The Frequency vs MonetaryValue KDE plot shows the highest density among low-frequency customers, but the density expands upward as Frequency increases, indicating that repeat purchasing is associated with higher spending."
    )
    lines.append(
        "The Recency vs Frequency KDE plot shows that higher-frequency customers are concentrated at lower Recency values, meaning that active customers tend to purchase more often."
    )
    lines.append(
        "The Recency vs MonetaryValue KDE plot indicates that higher-value customers are more concentrated among recently active customers, while customers with larger Recency values are mostly lower spenders."
    )
    lines.append("")

    lines.append("6. Overall Conclusion")
    lines.append("-" * 30)
    lines.append(
        "The statistical analysis consistently shows that customer value is positively associated with purchase frequency and negatively associated with recency."
    )
    lines.append(
        "These findings support the usefulness of RFM analysis for identifying valuable customer groups and for guiding dashboard storytelling in the final application."
    )

    return lines


# ============================================================
# Main
# ============================================================
def main() -> None:
    ensure_dir(OUTPUT_DIR)
    set_plot_theme()

    df = load_data()
    cols = get_numeric_columns(df)

    if len(cols) < 2:
        raise ValueError("At least two numeric RFM columns are required.")

    # Descriptive statistics
    stats_df = compute_statistics(df, cols)
    save_table(stats_df, "descriptive_statistics.csv")

    print("\nDescriptive Statistics:")
    print(tabulate(stats_df, headers="keys", tablefmt="grid"))

    # Correlations
    pearson_corr, pearson_p = correlation_with_pvalues(df, cols, method="pearson")
    spearman_corr, spearman_p = correlation_with_pvalues(df, cols, method="spearman")

    save_table(pearson_corr, "pearson_correlation.csv")
    save_table(pearson_p, "pearson_pvalues.csv")
    save_table(spearman_corr, "spearman_correlation.csv")
    save_table(spearman_p, "spearman_pvalues.csv")

    # Plots
    plot_heatmap(pearson_corr, "Pearson Correlation Heatmap", "pearson_heatmap.png")
    plot_heatmap(spearman_corr, "Spearman Correlation Heatmap", "spearman_heatmap.png")
    plot_scatter_matrix(df, cols, "scatter_matrix.png")

    kde_pairs = [
        ("Frequency", "MonetaryValue"),
        ("Recency", "MonetaryValue"),
        ("Recency", "Frequency"),
    ]

    for x_col, y_col in kde_pairs:
        if x_col in df.columns and y_col in df.columns:
            plot_multivariate_kde(df, x_col, y_col, f"kde_{x_col}_{y_col}.png")

    # Observations
    observation_lines = build_observations(stats_df, pearson_corr, spearman_corr)
    save_text(observation_lines, "observations.txt")

    print(f"\nObservations saved to: {os.path.join(OUTPUT_DIR, 'observations.txt')}")
    print("\nLayer 9 completed successfully.")


if __name__ == "__main__":
    main()