"""
04_normality_tests.py

Layer 5 — Normality Testing
Tests included:
- Shapiro-Wilk
- Kolmogorov-Smirnov
- D'Agostino K²

Includes:
- QQ plots
- Tabulated summary table
- Interpretation

Author: Syed Ibrahim Hamza
Project: E-Commerce Customer Behavior & RFM Analytics
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tabulate import tabulate

warnings.filterwarnings("ignore")


# =========================
# Configuration
# =========================
DATA_PATHS = [
    Path("data/rfm_table.csv"),
    Path("data/preprocessed_transactions.csv"),
    Path("data/processed_transactions.csv"),
]

OUTPUT_DIR = Path("phase1_static/normality_tests")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05

# Preferred columns for this layer
PREFERRED_COLUMNS = [
    "Recency",
    "Frequency",
    "MonetaryValue",
    "LineTotal",
    "Quantity",
    "Price",
]


# =========================
# Utility functions
# =========================
def load_data():
    """
    Load the first available dataset from expected project paths.
    """
    for path in DATA_PATHS:
        if path.exists():
            print(f"Loaded data from: {path}")
            return pd.read_csv(path)

    raise FileNotFoundError(
        "No input dataset found. Expected one of:\n"
        + "\n".join(str(p) for p in DATA_PATHS)
    )


def get_numeric_columns(df: pd.DataFrame):
    """
    Return numeric columns, prioritizing common project columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    selected = [col for col in PREFERRED_COLUMNS if col in numeric_cols]

    if not selected:
        selected = numeric_cols

    return selected


def clean_series(series: pd.Series):
    """
    Drop NaN and infinite values.
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s


def interpret_p_value(p_value, alpha=ALPHA):
    """
    Return normality interpretation from p-value.
    """
    if pd.isna(p_value):
        return "Test not valid"
    return "Looks Normal" if p_value > alpha else "Not Normal"


def sample_for_shapiro(series: pd.Series, max_n=5000, random_state=42):
    """
    Shapiro-Wilk is often recommended for <= 5000 observations.
    If larger, use a random sample for stability.
    """
    if len(series) > max_n:
        return series.sample(max_n, random_state=random_state)
    return series


# =========================
# Statistical tests
# =========================
def run_shapiro(series: pd.Series):
    """
    Shapiro-Wilk normality test.
    """
    try:
        s = sample_for_shapiro(series)
        stat, p = stats.shapiro(s)
        note = "Sampled to 5000" if len(series) > 5000 else "Full sample used"
        return stat, p, note
    except Exception as e:
        return np.nan, np.nan, f"Error: {e}"


def run_ks(series: pd.Series):
    """
    Kolmogorov-Smirnov test against fitted normal distribution.
    Data is standardized first.
    """
    try:
        mean = series.mean()
        std = series.std(ddof=1)

        if std == 0 or pd.isna(std):
            return np.nan, np.nan, "Zero variance"

        standardized = (series - mean) / std
        stat, p = stats.kstest(standardized, "norm")
        return stat, p, "Against fitted normal distribution"
    except Exception as e:
        return np.nan, np.nan, f"Error: {e}"


def run_dagostino(series: pd.Series):
    """
    D'Agostino and Pearson's K² normality test.
    Requires at least 8 observations.
    """
    try:
        if len(series) < 8:
            return np.nan, np.nan, "Requires at least 8 observations"
        stat, p = stats.normaltest(series)
        return stat, p, "Full sample used"
    except Exception as e:
        return np.nan, np.nan, f"Error: {e}"


# =========================
# Plotting
# =========================
def create_qq_plot(series: pd.Series, column_name: str, output_dir: Path):
    """
    Save QQ plot for a single variable.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    stats.probplot(series, dist="norm", plot=ax)
    ax.set_title(f"QQ Plot — {column_name}", fontsize=14)
    ax.grid(True, alpha=0.3)

    output_path = output_dir / f"qqplot_{column_name.lower()}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


# =========================
# Main workflow
# =========================
def normality_analysis(df: pd.DataFrame, columns: list[str]):
    """
    Run all normality tests for selected columns.
    """
    results = []

    for col in columns:
        series = clean_series(df[col])

        if len(series) < 3:
            print(f"Skipping {col}: not enough valid observations.")
            continue

        print("\n" + "=" * 80)
        print(f"NORMALITY TESTING FOR: {col}")
        print("=" * 80)

        # QQ plot
        qq_path = create_qq_plot(series, col, OUTPUT_DIR)

        # Tests
        shapiro_stat, shapiro_p, shapiro_note = run_shapiro(series)
        ks_stat, ks_p, ks_note = run_ks(series)
        dag_stat, dag_p, dag_note = run_dagostino(series)

        # Row-wise results
        results.append([
            col,
            len(series),
            "Shapiro-Wilk",
            round(shapiro_stat, 6) if pd.notna(shapiro_stat) else np.nan,
            round(shapiro_p, 6) if pd.notna(shapiro_p) else np.nan,
            interpret_p_value(shapiro_p),
            shapiro_note
        ])

        results.append([
            col,
            len(series),
            "Kolmogorov-Smirnov",
            round(ks_stat, 6) if pd.notna(ks_stat) else np.nan,
            round(ks_p, 6) if pd.notna(ks_p) else np.nan,
            interpret_p_value(ks_p),
            ks_note
        ])

        results.append([
            col,
            len(series),
            "D'Agostino K²",
            round(dag_stat, 6) if pd.notna(dag_stat) else np.nan,
            round(dag_p, 6) if pd.notna(dag_p) else np.nan,
            interpret_p_value(dag_p),
            dag_note
        ])

        # Console interpretation
        print(f"QQ plot saved to: {qq_path}")
        print(f"Observation:")
        print(
            f"- {col} appears "
            f"{'approximately normal' if (pd.notna(shapiro_p) and shapiro_p > ALPHA and pd.notna(ks_p) and ks_p > ALPHA and pd.notna(dag_p) and dag_p > ALPHA) else 'non-normal'} "
            f"based on the combined evidence from the tests."
        )

    return pd.DataFrame(
        results,
        columns=[
            "Feature",
            "N",
            "Test",
            "Statistic",
            "p-value",
            "Decision",
            "Notes"
        ]
    )


def generate_feature_summary(results_df: pd.DataFrame):
    """
    Create a feature-level overall interpretation.
    """
    summaries = []

    for feature in results_df["Feature"].unique():
        subset = results_df[results_df["Feature"] == feature]

        normal_votes = (subset["Decision"] == "Looks Normal").sum()
        non_normal_votes = (subset["Decision"] == "Not Normal").sum()

        if non_normal_votes >= 2:
            overall = "Overall evidence suggests the feature is not normally distributed."
        elif normal_votes >= 2:
            overall = "Overall evidence suggests the feature is approximately normal."
        else:
            overall = "Results are mixed across tests."

        summaries.append([feature, overall])

    return pd.DataFrame(summaries, columns=["Feature", "Overall Interpretation"])


def save_results(results_df: pd.DataFrame, summary_df: pd.DataFrame):
    """
    Save outputs to CSV and TXT.
    """
    csv_path = OUTPUT_DIR / "normality_test_results.csv"
    txt_path = OUTPUT_DIR / "normality_test_results.txt"
    summary_path = OUTPUT_DIR / "normality_summary.csv"

    results_df.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("NORMALITY TEST RESULTS\n")
        f.write("=" * 100 + "\n\n")
        f.write(tabulate(results_df, headers="keys", tablefmt="grid", showindex=False))
        f.write("\n\n")
        f.write("OVERALL INTERPRETATION\n")
        f.write("=" * 100 + "\n\n")
        f.write(tabulate(summary_df, headers="keys", tablefmt="grid", showindex=False))

    return csv_path, txt_path, summary_path


def main():
    df = load_data()
    columns = get_numeric_columns(df)

    print("\nSelected numeric columns for normality testing:")
    print(columns)

    results_df = normality_analysis(df, columns)

    if results_df.empty:
        print("No valid columns were available for testing.")
        return

    summary_df = generate_feature_summary(results_df)

    print("\n" + "=" * 100)
    print("TABULATED NORMALITY TEST RESULTS")
    print("=" * 100)
    print(tabulate(results_df, headers="keys", tablefmt="grid", showindex=False))

    print("\n" + "=" * 100)
    print("OVERALL INTERPRETATION")
    print("=" * 100)
    print(tabulate(summary_df, headers="keys", tablefmt="grid", showindex=False))

    csv_path, txt_path, summary_path = save_results(results_df, summary_df)

    print("\nSaved outputs:")
    print(f"- Detailed CSV: {csv_path}")
    print(f"- Summary CSV:  {summary_path}")
    print(f"- TXT Table:    {txt_path}")
    print(f"- QQ Plots:     {OUTPUT_DIR}")


if __name__ == "__main__":
    main()