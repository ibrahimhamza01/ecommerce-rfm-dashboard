import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import numpy as np
sns.set_theme(style="whitegrid")

INPUT_PATH = "data/processed_transactions.csv"
BASE_OUTPUT_DIR = "phase1_static/categorical_eda"


def create_subfolders():
    subfolders = {
        "countplots": os.path.join(BASE_OUTPUT_DIR, "countplots"),
        "barplots": os.path.join(BASE_OUTPUT_DIR, "barplots"),
        "stacked_barplots": os.path.join(BASE_OUTPUT_DIR, "stacked_barplots"),
        "grouped_barplots": os.path.join(BASE_OUTPUT_DIR, "grouped_barplots"),
        "piecharts": os.path.join(BASE_OUTPUT_DIR, "piecharts"),
        "heatmaps": os.path.join(BASE_OUTPUT_DIR, "heatmaps"),
        "clustermaps": os.path.join(BASE_OUTPUT_DIR, "clustermaps"),
        "stripplots": os.path.join(BASE_OUTPUT_DIR, "stripplots"),
        "swarmplots": os.path.join(BASE_OUTPUT_DIR, "swarmplots"),
    }

    for folder in subfolders.values():
        os.makedirs(folder, exist_ok=True)

    return subfolders


def load_data(path):
    df = pd.read_csv(path)
    return df


def plot_transaction_status_count(df, folders):
    plt.figure(figsize=(8, 5))

    order = df["TransactionStatus"].value_counts().index
    ax = sns.countplot(data=df, x="TransactionStatus", order=order)

    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)

    plt.title("Count Plot of Transaction Status", fontsize=14)
    plt.xlabel("Transaction Status")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    output_path = os.path.join(
        folders["countplots"],
        "countplot_transaction_status.png"
    )
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    completed = df[df["TransactionStatus"] == "Completed"].shape[0]
    cancelled = df[df["TransactionStatus"] == "Cancelled"].shape[0]

    total = len(df)
    completed_pct = (completed / total) * 100
    cancelled_pct = (cancelled / total) * 100

    print(f"Completed: {completed:,} ({completed_pct:.2f}%)")
    print(f"Cancelled: {cancelled:,} ({cancelled_pct:.2f}%)")

def plot_top_countries_by_revenue(df, folders, top_n=10):

    country_revenue = (
        df.groupby("Country")["LineTotal"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x=country_revenue.index,
        y=country_revenue.values
    )

    for i, value in enumerate(country_revenue.values):
        ax.text(
            i,
            value,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.title(f"Top {top_n} Countries by Total Revenue", fontsize=14)
    plt.xlabel("Country")
    plt.ylabel("Total Revenue")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

    output_path = os.path.join(
        folders["barplots"],
        "barplot_top_countries_revenue.png"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    top_country = country_revenue.idxmax()
    top_value = country_revenue.max()

    print(f"{top_country} has the highest total revenue at {top_value:,.2f}.")

def plot_grouped_bar_transaction_by_quarter(df, folders):

    df["PurchaseQuarterFormatted"] = df["PurchaseQuarter"].str.replace("Q", " Q")

    quarter_order = sorted(df["PurchaseQuarterFormatted"].unique())
    df["PurchaseQuarterFormatted"] = pd.Categorical(
        df["PurchaseQuarterFormatted"],
        categories=quarter_order,
        ordered=True
    )

    grouped_data = (
        df.groupby(["PurchaseQuarterFormatted", "TransactionStatus"])
        .size()
        .reset_index(name="Count")
    )

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=grouped_data,
        x="PurchaseQuarterFormatted",
        y="Count",
        hue="TransactionStatus"
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, fontsize=8)

    plt.title("Grouped Bar Plot of Transaction Status by Purchase Quarter", fontsize=14)
    plt.xlabel("Purchase Quarter")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Transaction Status")

    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

    output_path = os.path.join(
        folders["grouped_barplots"],
        "grouped_bar_transaction_status_by_quarter.png"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    top_row = grouped_data.loc[grouped_data["Count"].idxmax()]
    print("\nObservation:")
    print(
        f"The highest transaction count is in {top_row['PurchaseQuarterFormatted']} "
        f"for {top_row['TransactionStatus']} transactions at {top_row['Count']:,}."
    )

def plot_stacked_bar_transaction_by_quarter(df, folders):
    df["PurchaseQuarterFormatted"] = df["PurchaseQuarter"].str.replace("Q", " Q")

    quarter_order = sorted(df["PurchaseQuarterFormatted"].unique())
    df["PurchaseQuarterFormatted"] = pd.Categorical(
        df["PurchaseQuarterFormatted"],
        categories=quarter_order,
        ordered=True
    )

    stacked_data = pd.crosstab(
        df["PurchaseQuarterFormatted"],
        df["TransactionStatus"]
    )

    ax = stacked_data.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6)
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%d", fontsize=8)

    plt.title("Stacked Bar Plot of Transaction Status by Purchase Quarter", fontsize=14)
    plt.xlabel("Purchase Quarter")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Transaction Status")

    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

    output_path = os.path.join(
        folders["stacked_barplots"],
        "stacked_bar_transaction_status_by_quarter.png"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    quarter_totals = stacked_data.sum(axis=1)
    top_quarter = quarter_totals.idxmax()
    top_total = quarter_totals.max()

    print(f"{top_quarter} has the highest total transaction count at {top_total:,}.")

def plot_stacked_bar_price_category_by_quarter(df, folders):
    df["PurchaseQuarterFormatted"] = df["PurchaseQuarter"].str.replace("Q", " Q")

    quarter_order = sorted(df["PurchaseQuarterFormatted"].unique())
    df["PurchaseQuarterFormatted"] = pd.Categorical(
        df["PurchaseQuarterFormatted"],
        categories=quarter_order,
        ordered=True
    )

    stacked_data = pd.crosstab(
        df["PurchaseQuarterFormatted"],
        df["PriceCategory"]
    )

    ax = stacked_data.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6)
    )

    plt.title("Stacked Bar Plot of Price Category by Purchase Quarter", fontsize=14)
    plt.xlabel("Purchase Quarter")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Price Category", bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

    output_path = os.path.join(
        folders["stacked_barplots"],
        "stacked_bar_price_category_by_quarter.png"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_price_category_pie(df, folders):
    counts = df["PriceCategory"].value_counts()

    plt.figure(figsize=(6, 6))

    plt.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("Distribution of Price Categories")

    output_path = os.path.join(
        folders["piecharts"],
        "pie_price_category_distribution.png"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    top_category = counts.idxmax()
    top_value = counts.max()

    print(f"{top_category} is the most common price category with {top_value:,} transactions.")

def plot_heatmap_country_price_category(df, folders, top_n=10):
    top_countries = df["Country"].value_counts().head(top_n).index
    filtered_df = df[df["Country"].isin(top_countries)]

    heatmap_data = pd.crosstab(
        filtered_df["Country"],
        filtered_df["PriceCategory"]
    )

    heatmap_data = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu"
    )

    plt.title(f"Heatmap of Price Category by Country (Top {top_n})", fontsize=14)
    plt.xlabel("Price Category")
    plt.ylabel("Country")

    output_path = os.path.join(
        folders["heatmaps"],
        "heatmap_country_price_category.png"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    top_cell = heatmap_data.stack().idxmax()
    top_value = heatmap_data.stack().max()

    print(f"The highest concentration is {top_cell[0]} - {top_cell[1]} with {top_value:,} transactions.")

def plot_clustermap_country_quarter(df, folders, top_n=10):

    top_countries = df["Country"].value_counts().head(top_n).index
    filtered_df = df[df["Country"].isin(top_countries)]

    df_temp = filtered_df.copy()
    df_temp["PurchaseQuarterFormatted"] = df_temp["PurchaseQuarter"].str.replace("Q", " Q")

    cluster_data = pd.crosstab(
        df_temp["Country"],
        df_temp["PurchaseQuarterFormatted"]
    )

    cluster_data = cluster_data.div(cluster_data.sum(axis=1), axis=0)

    g = g = sns.clustermap(
        cluster_data,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        figsize=(12, 6),
        col_cluster=False
    )

    g.fig.suptitle("Cluster Map of Country Activity by Purchase Quarter", y=1.02)

    output_path = os.path.join(
        folders["clustermaps"],
        "clustermap_country_quarter.png"
    )
    g.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Countries are grouped based on similar purchasing activity patterns across quarters.")

def plot_strip_price_vs_revenue(df, folders):

    df = df.copy()
    df["LogLineTotal"] = np.log1p(df["LineTotal"].abs())

    sample_df = df.sample(min(3000, len(df)), random_state=42)

    plt.figure(figsize=(10, 6))
    sns.stripplot(
        data=sample_df,
        x="PriceCategory",
        y="LogLineTotal",
        jitter=True,
        alpha=0.5,
        order=["Very Low", "Low", "Medium", "High"]
    )

    plt.title("Strip Plot of Log(LineTotal) by Price Category", fontsize=14)
    plt.xlabel("Price Category")
    plt.ylabel("Log(LineTotal)")

    output_path = os.path.join(
        folders["stripplots"],
        "strip_pricecategory_loglinetotal.png"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("After log transformation, higher price categories show higher transaction values with clearer separation.")

def plot_strip_quarter_vs_quantity(df, folders):

    df = df.copy()
    df["LogQuantity"] = np.log1p(df["Quantity"].abs())

    sample_df = df.sample(min(3000, len(df)), random_state=42)
    sample_df["PurchaseQuarterFormatted"] = sample_df["PurchaseQuarter"].str.replace("Q", " Q")

    quarter_order = sorted(sample_df["PurchaseQuarterFormatted"].unique())
    sample_df["PurchaseQuarterFormatted"] = pd.Categorical(
        sample_df["PurchaseQuarterFormatted"],
        categories=quarter_order,
        ordered=True
    )

    plt.figure(figsize=(12, 6))
    sns.stripplot(
        data=sample_df,
        x="PurchaseQuarterFormatted",
        y="LogQuantity",
        jitter=True,
        alpha=0.5
    )

    plt.title("Strip Plot of Log(Quantity) by Purchase Quarter", fontsize=14)
    plt.xlabel("Purchase Quarter")
    plt.ylabel("Log(Quantity)")
    plt.xticks(rotation=45)

    output_path = os.path.join(
        folders["stripplots"],
        "strip_quarter_logquantity.png"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Log-transformed quantities reveal clearer distribution patterns across quarters, with variability present but no extreme dominance in any single period.")

def plot_dayofweek_quantity_box_swarm(df, folders, sample_size=1000):

    data = df.copy()
    data = data[data["Quantity"] > 0]
    data["LogQuantity"] = np.log1p(data["Quantity"])

    swarm_df = data.sample(min(sample_size, len(data)), random_state=42)

    day_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=data,
        x="DayOfWeek",
        y="LogQuantity",
        order=day_order,
        showfliers=False,
        palette="pastel"
    )

    sns.swarmplot(
        data=swarm_df,
        x="DayOfWeek",
        y="LogQuantity",
        order=day_order,
        size=1.5,
        color="black",
        alpha=0.6
    )

    plt.title("Distribution of Log(Quantity) by Day of Week", fontsize=14)
    plt.xlabel("Day of Week")
    plt.ylabel("Log(Quantity)")
    plt.xticks(rotation=30)

    output_path = os.path.join(
        folders["swarmplots"],
        "box_swarm_dayofweek_log_quantity.png"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(
        "Purchasing behavior is relatively consistent across the week, "
        "with minor variations in transaction quantity and dispersion."
    )


def main():
    folders = create_subfolders()
    df = load_data(INPUT_PATH)

    print("Columns in dataset:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:\n")
    print(df.head())

    print("\nGenerating Count plot...")
    plot_transaction_status_count(df, folders)

    print("\nGenerating Bar plot...")
    plot_top_countries_by_revenue(df, folders, top_n=10)

    print("\nGenerating Grouped Bar plot...")
    plot_grouped_bar_transaction_by_quarter(df, folders)

    print("\nGenerating Stacked Bar plot...")
    plot_stacked_bar_price_category_by_quarter(df, folders)

    print("\nGenerating Pie Chart...")
    plot_price_category_pie(df, folders)

    print("\nGenerating Heatmap...")
    plot_heatmap_country_price_category(df, folders)

    print("\nGenerating Cluster plot...")
    plot_clustermap_country_quarter(df, folders)

    print("\nGenerating Strip plot...")
    plot_strip_price_vs_revenue(df, folders)
    plot_strip_quarter_vs_quantity(df, folders)

    print("\nGenerating Swarm plot...")
    plot_dayofweek_quantity_box_swarm(df, folders)

if __name__ == "__main__":
    main()