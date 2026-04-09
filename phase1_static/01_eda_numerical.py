import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

DATA_PATH = "data/rfm_table.csv"
OUTPUT_DIR = "phase1_static/numerical_eda"

def load_data(path):
    df = pd.read_csv(path)
    return df


def create_output_folder(path):
    os.makedirs(path, exist_ok=True)

def get_numeric_columns(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "CustomerID"]
    return numeric_cols

def create_subfolders():
    hist_path = os.path.join(OUTPUT_DIR, "histogram_kde")
    dist_path = os.path.join(OUTPUT_DIR, "dist_plot")
    kde_path = os.path.join(OUTPUT_DIR, "kde_filled")
    box_path = os.path.join(OUTPUT_DIR, "boxplot")
    violin_path = os.path.join(OUTPUT_DIR, "violin")
    multibox_path = os.path.join(OUTPUT_DIR, "multivariate_boxplot")
    boxen_path = os.path.join(OUTPUT_DIR, "boxen_plot")
    scatter_path = os.path.join(OUTPUT_DIR, "scatter")
    reg_path = os.path.join(OUTPUT_DIR, "regression")
    joint_path = os.path.join(OUTPUT_DIR, "joint_plot")
    pairplot_path = os.path.join(OUTPUT_DIR, "pairplot")
    qq_path = os.path.join(OUTPUT_DIR, "qqplot")
    rug_path = os.path.join(OUTPUT_DIR, "rug")
    hexbin_path = os.path.join(OUTPUT_DIR, "hexbin")
    area_path = os.path.join(OUTPUT_DIR, "area")
    line_path = os.path.join(OUTPUT_DIR, "line")
    three_d_path = os.path.join(OUTPUT_DIR, "3d")
    contour_path = os.path.join(OUTPUT_DIR, "contour")

    os.makedirs(hist_path, exist_ok=True)
    os.makedirs(dist_path, exist_ok=True)
    os.makedirs(kde_path, exist_ok=True)
    os.makedirs(box_path, exist_ok=True)
    os.makedirs(violin_path, exist_ok=True)
    os.makedirs(multibox_path, exist_ok=True)
    os.makedirs(boxen_path, exist_ok=True)
    os.makedirs(scatter_path, exist_ok=True)
    os.makedirs(reg_path, exist_ok=True)
    os.makedirs(joint_path, exist_ok=True)
    os.makedirs(pairplot_path, exist_ok=True)
    os.makedirs(qq_path, exist_ok=True)
    os.makedirs(rug_path, exist_ok=True)
    os.makedirs(hexbin_path, exist_ok=True)
    os.makedirs(area_path, exist_ok=True)
    os.makedirs(line_path, exist_ok=True)
    os.makedirs(three_d_path, exist_ok=True)
    os.makedirs(contour_path, exist_ok=True)

    return (hist_path, dist_path, kde_path, box_path, violin_path, multibox_path,
            boxen_path, scatter_path, reg_path, joint_path, pairplot_path, qq_path,
            rug_path, hexbin_path, area_path, line_path, three_d_path, contour_path)

def plot_histogram_kde(df, col, save_dir):
    plt.figure(figsize=(10, 6))

    sns.histplot(df[col].dropna(), kde=True, bins=30)

    plt.title(f"Histogram + KDE of {col}", fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(save_dir, f"{col}_hist_kde.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # Observation
    skewness = df[col].dropna().skew()
    mean_val = df[col].mean()
    median_val = df[col].median()

    if skewness > 1:
        obs = f"{col} is highly right-skewed, with most values concentrated at the lower range and a long tail toward higher values."
    elif skewness < -1:
        obs = f"{col} is highly left-skewed, with most values concentrated at the higher range and a long tail toward lower values."
    elif mean_val > median_val:
        obs = f"{col} shows mild positive skewness, suggesting some higher-value observations are pulling the distribution to the right."
    elif mean_val < median_val:
        obs = f"{col} shows mild negative skewness, suggesting some lower-value observations are pulling the distribution to the left."
    else:
        obs = f"{col} appears fairly symmetric overall."

def plot_distplot(df, col, save_dir):
    plt.figure(figsize=(10, 6))

    sns.histplot(df[col].dropna(), kde=True, stat="density", bins=30)

    plt.title(f"Dist Plot of {col}", fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(save_dir, f"{col}_distplot.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    std_val = df[col].std()

    obs = f"{col} shows a spread of values with standard deviation = {std_val:.2f}, indicating the variability of the distribution."
    print(f"[Dist Plot] {col}: {obs}")

def plot_kde_filled(df, col, save_dir):
    plt.figure(figsize=(10, 6))

    sns.kdeplot(df[col].dropna(), fill=True)

    plt.title(f"Filled KDE Plot of {col}", fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(save_dir, f"{col}_kde_filled.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # Observation
    peak = df[col].mode().iloc[0]

    obs = f"{col} shows highest density around {peak}, indicating concentration of values near this region."
    print(f"[KDE Filled] {col}: {obs}")

def plot_boxplot(df, col, save_dir):
    plt.figure(figsize=(8, 6))

    sns.boxplot(y=df[col], color="skyblue")

    plt.title(f"Boxplot of {col}", fontsize=14)
    plt.xlabel("All Customers")
    plt.ylabel(col)
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(save_dir, f"{col}_boxplot.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # Outlier detection
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]

    obs = f"{col} contains {outliers} potential outliers based on IQR method."
    print(f"[Boxplot] {col}: {obs}")

def plot_violin(df, col, save_dir):
    plt.figure(figsize=(8, 6))

    sns.violinplot(y=df[col], inner="quartile")

    plt.title(f"Violin Plot of {col}", fontsize=14)
    plt.xlabel("Distribution")
    plt.ylabel(col)
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(save_dir, f"{col}_violin.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    median = df[col].median()

    obs = f"{col} distribution is centered around median value {median:.2f}, with visible density spread."
    print(f"[Violin Plot] {col}: {obs}")

def create_recency_bins(df):
    df["RecencySegment"] = pd.qcut(
        df["Recency"],
        q=3,
        labels=["Low", "Medium", "High"]
    )
    return df

def plot_multivariate_boxplot(df, save_dir):
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="RecencySegment",
        y="Frequency",
        order=["Low", "Medium", "High"],
        palette="Set2"
    )

    plt.title("Multivariate Boxplot of Frequency by Recency Segment", fontsize=14)
    plt.xlabel("Recency Segment")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(save_dir, "frequency_by_recencysegment_boxplot.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    medians = df.groupby("RecencySegment")["Frequency"].median().reindex(["Low", "Medium", "High"])

    obs = (
        f"Median Frequency by segment -> "
        f"Low: {medians['Low']:.2f}, "
        f"Medium: {medians['Medium']:.2f}, "
        f"High: {medians['High']:.2f}."
    )
    print(f"[Multivariate Boxplot] {obs}")

def plot_multivariate_boxen(df, save_dir):
    plt.figure(figsize=(10, 6))

    sns.boxenplot(
        data=df,
        x="RecencySegment",
        y="Frequency",
        order=["Low", "Medium", "High"],
        palette="Set2"
    )

    plt.title("Boxen Plot of Frequency by Recency Segment", fontsize=14)
    plt.xlabel("Recency Segment")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(save_dir, "frequency_by_recencysegment_boxen.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print("[Boxen Plot] Shows deeper distribution across quantiles for each segment.")

def plot_scatter(df, x_col, y_col, save_dir, jitter_x=False, jitter_y=False):
    plt.figure(figsize=(8, 6))

    x = df[x_col]
    y = df[y_col]

    # Apply jitter if requested
    if jitter_x:
        x = x + np.random.normal(0, 5, size=len(df))

    if jitter_y:
        y = y + np.random.normal(0, 0.1, size=len(df))

    sns.scatterplot(
        x=x,
        y=y,
        alpha=0.3,
        s=20
    )

    plt.title(f"{y_col} vs {x_col}", fontsize=14)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(save_dir, f"{y_col}_vs_{x_col}_scatter.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    corr = df[[x_col, y_col]].corr().iloc[0, 1]

    obs = f"There is a correlation of {corr:.2f} between {x_col} and {y_col}."
    print(f"[Scatter] {y_col} vs {x_col}: {obs}")

def plot_regression(df, x_col, y_col, save_dir):
    plt.figure(figsize=(8, 6))

    sns.regplot(
        data=df,
        x=x_col,
        y=y_col,
        scatter_kws={"alpha": 0.3, "s": 20},
        line_kws={"color": "red"}
    )

    plt.title(f"Regression Plot: {y_col} vs {x_col}", fontsize=14)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(save_dir, f"{y_col}_vs_{x_col}_regplot.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    corr = df[[x_col, y_col]].corr().iloc[0, 1]

    if corr > 0:
        direction = "positive"
    elif corr < 0:
        direction = "negative"
    else:
        direction = "no"

    obs = f"A {direction} relationship is observed with correlation {corr:.2f}."
    print(f"[Regression] {y_col} vs {x_col}: {obs}")

def plot_joint(df, x_col, y_col, save_dir):
    joint = sns.jointplot(
        data=df,
        x=x_col,
        y=y_col,
        kind="hex",
        height=8,
        space=0,
        alpha=1
    )

    joint.fig.suptitle(f"Joint Plot: {y_col} vs {x_col}", fontsize=14)
    joint.fig.tight_layout()
    joint.fig.subplots_adjust(top=0.95)

    save_path = os.path.join(save_dir, f"{y_col}_vs_{x_col}_joint.png")
    joint.fig.savefig(save_path)
    plt.close()

    print(f"[Joint Plot] Generated for {y_col} vs {x_col}")

def plot_pairplot(df, save_dir):
    pair = sns.pairplot(
        df[["Recency", "Frequency", "MonetaryValue"]],
        diag_kind="kde",
        corner=False
    )

    pair.fig.suptitle("Pair Plot of RFM Variables", y=1.02, fontsize=16)

    save_path = os.path.join(save_dir, "rfm_pairplot.png")
    pair.fig.savefig(save_path, bbox_inches="tight")
    plt.close()

    print("[Pair Plot] Generated for Recency, Frequency, and MonetaryValue.")

def plot_qq(df, col, save_dir):
    plt.figure(figsize=(6, 6))

    qqplot(df[col].dropna(), line='s')

    plt.title(f"QQ Plot of {col}")
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(save_dir, f"{col}_qqplot.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[QQ Plot] {col}: shows deviation from normal distribution.")

def plot_rug(df, col, save_dir):
    plt.figure(figsize=(8, 4))

    sns.rugplot(x=df[col])

    plt.title(f"Rug Plot of {col}")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(save_dir, f"{col}_rug.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[Rug Plot] {col}: shows data point concentration.")

def plot_hexbin(df, x_col, y_col, save_dir):
    plt.figure(figsize=(8, 6))

    plt.hexbin(df[x_col], df[y_col], gridsize=30)

    plt.title(f"Hexbin Plot: {y_col} vs {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.colorbar(label="Count")

    save_path = os.path.join(save_dir, f"{y_col}_vs_{x_col}_hexbin.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[Hexbin] {y_col} vs {x_col}")

def plot_area(df, col, save_dir):
    plt.figure(figsize=(8, 6))

    sorted_vals = df[col].sort_values().reset_index(drop=True)

    plt.fill_between(sorted_vals.index, sorted_vals)

    plt.title(f"Area Plot of {col}")
    plt.xlabel("Sorted Observations")
    plt.ylabel(col)

    save_path = os.path.join(save_dir, f"{col}_area.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[Area Plot] {col}")

def plot_line(df, col, save_dir):
    plt.figure(figsize=(8, 6))

    sorted_vals = df[col].sort_values().reset_index(drop=True)

    plt.plot(sorted_vals)

    plt.title(f"Line Plot of {col}")
    plt.xlabel("Sorted Observations")
    plt.ylabel(col)

    save_path = os.path.join(save_dir, f"{col}_line.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[Line Plot] {col}")

def plot_3d(df, save_dir):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        df["Recency"],
        df["Frequency"],
        df["MonetaryValue"],
        alpha=0.3
    )

    ax.set_xlabel("Recency")
    ax.set_ylabel("Frequency")
    ax.set_zlabel("MonetaryValue")
    ax.set_title("3D Plot of RFM")

    save_path = os.path.join(save_dir, "rfm_3d.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print("[3D Plot] RFM relationship visualized.")

def plot_contour(df, save_dir):
    plt.figure(figsize=(8, 6))

    sns.kdeplot(
        x=df["Recency"],
        y=df["MonetaryValue"],
        fill=True,
        cmap="viridis"
    )

    plt.title("Contour Plot: MonetaryValue vs Recency")
    plt.xlabel("Recency")
    plt.ylabel("MonetaryValue")

    save_path = os.path.join(save_dir, "recency_vs_monetary_contour.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print("[Contour Plot] Density of Monetary vs Recency.")

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    create_output_folder(OUTPUT_DIR)

    numeric_cols = get_numeric_columns(df)
    (hist_path, dist_path, kde_path, box_path,
     violin_path, multibox_path,boxen_path,
     scatter_path, reg_path, joint_path, pairplot_path,
     qq_path, rug_path, hexbin_path, area_path, line_path,
     three_d_path, contour_path)= create_subfolders()

    print("Data loaded successfully.")
    print("Shape:", df.shape)

    print("\nNumeric columns for EDA:")
    print(numeric_cols)

    print("\nGenerating Histogram + KDE plots...\n")

    for col in numeric_cols:
        plot_histogram_kde(df, col, hist_path)

    print("\nGenerating Dist plots...\n")

    for col in numeric_cols:
        plot_distplot(df, col, dist_path)

    print("\nGenerating Filled KDE plots...\n")

    for col in numeric_cols:
        plot_kde_filled(df, col, kde_path)

    print("\nGenerating Boxplots...\n")

    for col in numeric_cols:
        plot_boxplot(df, col, box_path)

    print("\nGenerating Violin plots...\n")

    for col in numeric_cols:
        plot_violin(df, col, violin_path)

    df = create_recency_bins(df)
    print("\nRecencySegment created:")
    print(df["RecencySegment"].value_counts())

    print("\nGenerating Multivariate Boxplot...\n")
    plot_multivariate_boxplot(df, multibox_path)

    print("\nGenerating Multivariate Boxen Plot...\n")
    plot_multivariate_boxen(df, boxen_path)

    print("\nGenerating Scatter Plots...\n")
    plot_scatter(df, "Recency", "Frequency", scatter_path, jitter_y=True)
    plot_scatter(df, "Frequency", "MonetaryValue", scatter_path)
    plot_scatter(df, "Recency", "MonetaryValue", scatter_path, jitter_x=True)

    print("\nGenerating Regression Plots...\n")
    plot_regression(df, "Recency", "Frequency", reg_path)
    plot_regression(df, "Frequency", "MonetaryValue", reg_path)
    plot_regression(df, "Recency", "MonetaryValue", reg_path)

    print("\nGenerating Joint Plots...\n")
    plot_joint(df, "Recency", "Frequency", joint_path)
    plot_joint(df, "Frequency", "MonetaryValue", joint_path)
    plot_joint(df, "Recency", "MonetaryValue", joint_path)

    print("\nGenerating Pair Plot...\n")
    plot_pairplot(df, pairplot_path)

    print("\nGenerating QQ Plot...\n")
    for col in numeric_cols:
        plot_qq(df, col, qq_path)

    print("\nGenerating Rug Plot...\n")
    for col in numeric_cols:
        plot_rug(df, col, rug_path)

    print("\nGenerating Hexbin Plot...\n")
    plot_hexbin(df, "Recency", "Frequency", hexbin_path)
    plot_hexbin(df, "Frequency", "MonetaryValue", hexbin_path)
    plot_hexbin(df, "Recency", "MonetaryValue", hexbin_path)

    print("\nGenerating Area Plot...\n")
    for col in numeric_cols:
        plot_area(df, col, area_path)

    print("\n Generating Line Plot...\n")
    for col in numeric_cols:
        plot_line(df, col, line_path)

    print("\nGenerating 3d Plot...\n")
    plot_3d(df, three_d_path)

    print("\nGenerating Contour Plot...\n")
    plot_contour(df, contour_path)
