import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tabulate import tabulate


# -----------------------------
# CONFIG
# -----------------------------
INPUT_PATH = "data/processed_transactions.csv"
OUTPUT_DIR = "phase1_static/transformation_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set(style="whitegrid", palette="deep")


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(path=INPUT_PATH):
    return pd.read_csv(path)


# -----------------------------
# DATA PREP
# -----------------------------
def prepare_positive_series(df, feature):
    """
    Use only strictly positive values for transformation analysis.
    This avoids mixing purchases with returns/cancellations/adjustments.
    """
    s = pd.to_numeric(df[feature], errors="coerce").dropna().astype(float)
    s = s[np.isfinite(s)]
    s = s[s > 0]

    if len(s) < 8:
        raise ValueError(f"{feature}: not enough strictly positive values for analysis.")

    return s.reset_index(drop=True)


def trim_for_visual(series, lower_q=0.01, upper_q=0.99):
    """
    Trim only for plotting so extreme tails do not dominate the histogram.
    """
    s = pd.Series(series).dropna().astype(float)
    lower = s.quantile(lower_q)
    upper = s.quantile(upper_q)
    return s[(s >= lower) & (s <= upper)]


# -----------------------------
# NORMALITY TESTING (RE-IMPLEMENTED HERE)
# -----------------------------
def run_normality_tests(series, sample_size=5000):
    """
    Re-run normality tests inside Layer 6 for before/after comparison.

    Tests:
    - Shapiro-Wilk
    - Kolmogorov-Smirnov
    - D'Agostino K^2

    Also returns:
    - Skewness
    - Kurtosis
    """
    s = pd.Series(series).dropna().astype(float)
    s = s[np.isfinite(s)]

    results = {
        "Shapiro_p": np.nan,
        "KS_p": np.nan,
        "Dagostino_p": np.nan,
        "Skewness": np.nan,
        "Kurtosis": np.nan,
    }

    if len(s) < 8:
        return results

    sample = s.sample(min(len(s), sample_size), random_state=42)

    try:
        results["Shapiro_p"] = stats.shapiro(sample)[1]
    except Exception:
        pass

    try:
        std = sample.std(ddof=0)
        if std > 0 and np.isfinite(std):
            z = (sample - sample.mean()) / std
            results["KS_p"] = stats.kstest(z, "norm")[1]
    except Exception:
        pass

    try:
        results["Dagostino_p"] = stats.normaltest(sample)[1]
    except Exception:
        pass

    try:
        results["Skewness"] = stats.skew(sample, bias=False)
    except Exception:
        pass

    try:
        results["Kurtosis"] = stats.kurtosis(sample, fisher=True, bias=False)
    except Exception:
        pass

    return results


# -----------------------------
# TRANSFORMATIONS
# -----------------------------
def log_transform(series):
    s = pd.Series(series).astype(float)
    transformed = np.log1p(s)
    return pd.Series(transformed, index=s.index), "log1p(x)"


def boxcox_transform(series):
    s = pd.Series(series).astype(float)

    if (s <= 0).any():
        raise ValueError("Box-Cox requires strictly positive values.")

    transformed, lam = stats.boxcox(s)
    return pd.Series(transformed, index=s.index), f"boxcox(x), lambda={lam:.4f}"


def standardize_transform(series):
    s = pd.Series(series).astype(float).values.reshape(-1, 1)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(s).flatten()
    return pd.Series(transformed), "standardization"


def minmax_transform(series):
    s = pd.Series(series).astype(float).values.reshape(-1, 1)
    scaler = MinMaxScaler()
    transformed = scaler.fit_transform(s).flatten()
    return pd.Series(transformed), "minmax_scaling"


# -----------------------------
# PLOTTING
# -----------------------------
def plot_before_after(original, transformed_dict, feature_name):
    methods = list(transformed_dict.keys())
    total_plots = 1 + len(methods)

    fig, axes = plt.subplots(1, total_plots, figsize=(5 * total_plots, 4))

    original_plot = trim_for_visual(original)
    sns.histplot(original_plot, kde=True, bins=50, ax=axes[0])
    axes[0].set_title(f"{feature_name} - Original")
    axes[0].set_xlabel(feature_name)
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)

    for i, method in enumerate(methods, start=1):
        data_plot = trim_for_visual(transformed_dict[method])
        sns.histplot(data_plot, kde=True, bins=50, ax=axes[i])
        axes[i].set_title(f"{feature_name} - {method}")
        axes[i].set_xlabel(method)
        axes[i].set_ylabel("Count")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{feature_name}_before_after.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# SCORING
# -----------------------------
def compute_score(row):
    """
    Rank transformations using:
    - higher average p-value
    - lower absolute skewness
    - lower absolute kurtosis
    """
    pvals = [row["Shapiro_p"], row["KS_p"], row["Dagostino_p"]]
    pvals = [p for p in pvals if pd.notna(p)]
    avg_p = np.mean(pvals) if pvals else np.nan

    skew_score = 1 / (1 + abs(row["Skewness"])) if pd.notna(row["Skewness"]) else np.nan
    kurt_score = 1 / (1 + abs(row["Kurtosis"])) if pd.notna(row["Kurtosis"]) else np.nan

    components = []

    if pd.notna(avg_p):
        components.append(0.5 * avg_p)
    if pd.notna(skew_score):
        components.append(0.3 * skew_score)
    if pd.notna(kurt_score):
        components.append(0.2 * kurt_score)

    return np.sum(components) if components else np.nan


# -----------------------------
# FEATURE ANALYSIS
# -----------------------------
def analyze_feature(df, feature):
    original = prepare_positive_series(df, feature)

    transformed_data = {}
    notes = {}
    results = []

    # Original + re-implemented normality testing
    original_tests = run_normality_tests(original)
    results.append({
        "Feature": feature,
        "Method": "Original",
        **original_tests
    })

    # Log
    try:
        log_s, note = log_transform(original)
        transformed_data["Log"] = log_s
        notes["Log"] = note
        results.append({
            "Feature": feature,
            "Method": "Log",
            **run_normality_tests(log_s)
        })
    except Exception as e:
        notes["Log"] = f"Failed: {e}"

    # Box-Cox
    try:
        boxcox_s, note = boxcox_transform(original)
        transformed_data["Box-Cox"] = boxcox_s
        notes["Box-Cox"] = note
        results.append({
            "Feature": feature,
            "Method": "Box-Cox",
            **run_normality_tests(boxcox_s)
        })
    except Exception as e:
        notes["Box-Cox"] = f"Failed: {e}"

    # Standardization
    try:
        std_s, note = standardize_transform(original)
        transformed_data["Standardization"] = std_s
        notes["Standardization"] = note
        results.append({
            "Feature": feature,
            "Method": "Standardization",
            **run_normality_tests(std_s)
        })
    except Exception as e:
        notes["Standardization"] = f"Failed: {e}"

    # MinMax
    try:
        mm_s, note = minmax_transform(original)
        transformed_data["MinMax"] = mm_s
        notes["MinMax"] = note
        results.append({
            "Feature": feature,
            "Method": "MinMax",
            **run_normality_tests(mm_s)
        })
    except Exception as e:
        notes["MinMax"] = f"Failed: {e}"

    plot_before_after(original, transformed_data, feature)

    results_df = pd.DataFrame(results)
    results_df["Avg_p_value"] = results_df[["Shapiro_p", "KS_p", "Dagostino_p"]].mean(axis=1, skipna=True)
    results_df["Abs_Skewness"] = results_df["Skewness"].abs()
    results_df["Abs_Kurtosis"] = results_df["Kurtosis"].abs()
    results_df["Score"] = results_df.apply(compute_score, axis=1)

    best_row = results_df.sort_values(
        ["Score", "Abs_Skewness", "Abs_Kurtosis"],
        ascending=[False, True, True]
    ).iloc[0]

    return results_df, notes, best_row, len(original)


# -----------------------------
# OBSERVATIONS
# -----------------------------
def generate_observation(feature, best_row, n_used):
    return (
        f"{feature}: Transformation analysis used {n_used:,} strictly positive records. "
        f"{best_row['Method']} performed best overall. "
        f"Skewness after transformation = {best_row['Skewness']:.4f}, "
        f"kurtosis = {best_row['Kurtosis']:.4f}. "
        f"Normality tests were re-run after transformation for direct before/after comparison."
    )


# -----------------------------
# MAIN
# -----------------------------
def main():
    df = load_data()

    candidate_features = [
        "Quantity", "Price", "LineTotal",
        "Recency", "Frequency", "MonetaryValue"
    ]
    features = [col for col in candidate_features if col in df.columns]

    if not features:
        raise ValueError("None of the expected numeric features were found in the dataset.")

    all_results = []
    observations = []

    print("\n" + "=" * 80)
    print("LAYER 6 — DATA TRANSFORMATION")
    print("=" * 80)

    for feature in features:
        print(f"\nProcessing feature: {feature}")

        try:
            results_df, notes, best_row, n_used = analyze_feature(df, feature)
            all_results.append(results_df)

            print(f"\nRecords used for transformation ({feature} > 0): {n_used:,}")

            print("\nTransformation notes:")
            for method, note in notes.items():
                print(f"- {method}: {note}")

            print("\nBefore/After Normality Comparison:")
            print(tabulate(results_df.round(6), headers="keys", tablefmt="grid", showindex=False))

            obs = generate_observation(feature, best_row, n_used)
            observations.append(obs)

            print("\nObservation:")
            print(obs)

        except Exception as e:
            print(f"\nSkipping {feature}: {e}")

    if not all_results:
        raise ValueError("No features were successfully transformed.")

    final_results = pd.concat(all_results, ignore_index=True)
    final_results.to_csv(
        os.path.join(OUTPUT_DIR, "transformation_results.csv"),
        index=False
    )

    best_summary = (
        final_results.sort_values(
            ["Score", "Abs_Skewness", "Abs_Kurtosis"],
            ascending=[False, True, True]
        )
        .groupby("Feature", as_index=False)
        .first()[[
            "Feature", "Method", "Avg_p_value",
            "Skewness", "Kurtosis",
            "Abs_Skewness", "Abs_Kurtosis", "Score"
        ]]
    )
    best_summary.to_csv(
        os.path.join(OUTPUT_DIR, "best_transformation_summary.csv"),
        index=False
    )

    print("\n" + "=" * 80)
    print("BEST METHOD SUMMARY")
    print("=" * 80)
    print(tabulate(best_summary.round(6), headers="keys", tablefmt="grid", showindex=False))

    print("\nFINAL CONCLUSION:")
    print(
        "Normality testing was re-implemented inside Layer 6 so each feature could be evaluated "
        "before and after transformation. Log and Box-Cox typically improve symmetry much more "
        "than Standardization and MinMax scaling, while scaling methods remain useful for later PCA and clustering."
    )

    with open(os.path.join(OUTPUT_DIR, "observations.txt"), "w", encoding="utf-8") as f:
        for obs in observations:
            f.write(obs + "\n\n")


if __name__ == "__main__":
    main()