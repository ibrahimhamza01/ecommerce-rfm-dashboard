# 07_subplots_storytelling.py

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10


# =========================================================
# Helper Functions
# =========================================================

def ensure_output_dir():
    """
    Create output folder for Layer 8 inside phase1_static/subplots.
    """
    output_dir = "phase1_static/subplots"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_data():
    """
    Load preprocessed transaction-level data and customer-level RFM data.
    """
    transactions_path = "data/processed_transactions.csv"
    rfm_path = "data/rfm_table.csv"

    if not os.path.exists(transactions_path):
        raise FileNotFoundError(f"Missing file: {transactions_path}")

    if not os.path.exists(rfm_path):
        raise FileNotFoundError(f"Missing file: {rfm_path}")

    df = pd.read_csv(transactions_path)
    rfm = pd.read_csv(rfm_path)

    # Convert possible date columns to datetime
    for col in ["InvoiceDate", "invoice_date", "Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            break

    return df, rfm


def standardize_columns(df, rfm):
    """
    Resolve possible column name variations across files.
    """
    tx_col_map = {
        "Country": ["Country", "country"],
        "LineTotal": ["LineTotal", "line_total", "Revenue", "revenue"],
        "TransactionStatus": ["TransactionStatus", "transaction_status", "Status"],
        "PurchaseQuarter": ["PurchaseQuarter", "purchase_quarter", "Quarter"],
        "InvoiceDate": ["InvoiceDate", "invoice_date", "Date"],
    }

    rfm_col_map = {
        "Recency": ["Recency", "recency"],
        "Frequency": ["Frequency", "frequency"],
        "MonetaryValue": ["MonetaryValue", "Monetary", "monetary", "Monetary_Value"],
    }

    resolved_tx = {}
    for standard_name, candidates in tx_col_map.items():
        for candidate in candidates:
            if candidate in df.columns:
                resolved_tx[standard_name] = candidate
                break

    resolved_rfm = {}
    for standard_name, candidates in rfm_col_map.items():
        for candidate in candidates:
            if candidate in rfm.columns:
                resolved_rfm[standard_name] = candidate
                break

    return resolved_tx, resolved_rfm


def create_missing_features(df, tx_cols):
    """
    Create or standardize Layer 8 plotting features.
    """
    invoice_date_col = tx_cols.get("InvoiceDate")

    if invoice_date_col is not None:
        df[invoice_date_col] = pd.to_datetime(df[invoice_date_col], errors="coerce")

        # Always standardize PurchaseQuarter to Q1, Q2, Q3, Q4
        df["PurchaseQuarter"] = df[invoice_date_col].dt.quarter.map(
            lambda x: f"Q{int(x)}" if pd.notna(x) else None
        )
        tx_cols["PurchaseQuarter"] = "PurchaseQuarter"

    return df, tx_cols


def print_observation(title, text):
    print("\n" + "=" * 80)
    print(f"OBSERVATION: {title}")
    print("-" * 80)
    print(text.strip())
    print("=" * 80 + "\n")


def save_observations(output_dir, observations):
    """
    Save all observations and combined narrative to a text file.
    """
    file_path = os.path.join(output_dir, "observations.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        for section_title, section_text in observations:
            f.write(section_title + "\n")
            f.write("-" * len(section_title) + "\n")
            f.write(section_text.strip() + "\n\n")
            f.write("=" * 80 + "\n\n")

    print(f"Observations saved to: {file_path}")


# =========================================================
# Figure 1 — Customer Behavior Story
# =========================================================

def plot_customer_behavior_story(rfm, rfm_cols, output_dir):
    recency_col = rfm_cols["Recency"]
    frequency_col = rfm_cols["Frequency"]
    monetary_col = rfm_cols["MonetaryValue"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Layer 8 Storytelling Figure 1: Customer Behavior Through RFM", fontsize=16)

    # Plot 1: Recency Distribution
    sns.histplot(rfm[recency_col].dropna(), kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Recency Distribution")
    axes[0, 0].set_xlabel("Recency (Days)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Frequency Distribution
    sns.histplot(rfm[frequency_col].dropna(), kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Frequency Distribution")
    axes[0, 1].set_xlabel("Frequency")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Monetary Value Distribution
    sns.histplot(rfm[monetary_col].dropna(), kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Monetary Value Distribution")
    axes[1, 0].set_xlabel("Monetary Value")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Recency vs Frequency
    sns.scatterplot(
        data=rfm,
        x=recency_col,
        y=frequency_col,
        alpha=0.6,
        ax=axes[1, 1]
    )
    axes[1, 1].set_title("Recency vs Frequency")
    axes[1, 1].set_xlabel("Recency (Days)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(
        os.path.join(output_dir, "figure1_customer_behavior.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    obs1 = """
Figure 1 combines the three core RFM variables with a relationship plot to tell a customer
behavior story. Recency is right-skewed, suggesting that many customers purchased relatively
recently while a long tail of customers has been inactive for much longer periods.

Frequency is also highly skewed, showing that most customers purchase only a few times,
while a smaller group purchases repeatedly. Monetary Value follows the same pattern, where
most customers contribute modest spending and a limited subset contributes much higher value.

The Recency vs Frequency scatter indicates that customers with lower recency values tend to
appear more engaged, while many high-recency customers buy less frequently. Overall, the
figure shows that customer value is unevenly distributed and that a smaller group of loyal
customers likely drives a disproportionate share of business performance.
"""
    print_observation("Figure 1 — Customer Behavior Story", obs1)
    return ("Figure 1 — Customer Behavior Story", obs1)


# =========================================================
# Figure 2 — Business / Transaction Story
# =========================================================

def plot_business_story(df, tx_cols, output_dir):
    revenue_col = tx_cols.get("LineTotal")
    country_col = tx_cols.get("Country")
    status_col = tx_cols.get("TransactionStatus")
    quarter_col = tx_cols.get("PurchaseQuarter")
    invoice_date_col = tx_cols.get("InvoiceDate")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Layer 8 Storytelling Figure 2: Transaction and Sales Patterns", fontsize=16)

    # Plot 1: Monthly Revenue Trend
    if revenue_col and invoice_date_col:
        monthly_revenue = (
            df.dropna(subset=[invoice_date_col])
            .assign(Month=df[invoice_date_col].dt.to_period("M").astype(str))
            .groupby("Month")[revenue_col]
            .sum()
            .reset_index()
        )

        sns.lineplot(data=monthly_revenue, x="Month", y=revenue_col, marker="o", ax=axes[0, 0])
        axes[0, 0].set_title("Monthly Revenue Trend")
        axes[0, 0].set_xlabel("Month")
        axes[0, 0].set_ylabel("Revenue")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, "Monthly revenue plot unavailable", ha="center", va="center")
        axes[0, 0].set_title("Monthly Revenue Trend")

    # Plot 2: Top 10 Countries by Revenue
    if revenue_col and country_col:
        country_revenue = (
            df.groupby(country_col)[revenue_col]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        sns.barplot(data=country_revenue, x=revenue_col, y=country_col, ax=axes[0, 1])
        axes[0, 1].set_title("Top 10 Countries by Revenue")
        axes[0, 1].set_xlabel("Revenue")
        axes[0, 1].set_ylabel("Country")
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, "Country revenue plot unavailable", ha="center", va="center")
        axes[0, 1].set_title("Top 10 Countries by Revenue")

    # Plot 3: Transaction Status Distribution
    if status_col:
        status_counts = df[status_col].value_counts().reset_index()
        status_counts.columns = [status_col, "Count"]

        sns.barplot(data=status_counts, x=status_col, y="Count", ax=axes[1, 0])
        axes[1, 0].set_title("Transaction Status Distribution")
        axes[1, 0].set_xlabel("Transaction Status")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, "Transaction status plot unavailable", ha="center", va="center")
        axes[1, 0].set_title("Transaction Status Distribution")

    # Plot 4: Purchase Quarter Distribution
    if quarter_col and quarter_col in df.columns and df[quarter_col].notna().sum() > 0:
        quarter_order = ["Q1", "Q2", "Q3", "Q4"]
        quarter_counts = (
            df[quarter_col]
            .value_counts()
            .reindex(quarter_order, fill_value=0)
            .reset_index()
        )
        quarter_counts.columns = [quarter_col, "Count"]

        sns.barplot(data=quarter_counts, x=quarter_col, y="Count", ax=axes[1, 1])
        axes[1, 1].set_title("Purchase Quarter Distribution")
        axes[1, 1].set_xlabel("Quarter")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(
        os.path.join(output_dir, "figure2_business_story.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    obs2 = """
Figure 2 shifts the narrative from customer-level behavior to business-level transaction
patterns. The monthly revenue trend shows clear variation across time, suggesting seasonality,
periods of stronger performance, and possible cyclical buying behavior.

The country-level revenue chart reveals strong geographic concentration, with the United Kingdom
contributing the overwhelming majority of revenue compared with all other countries. This suggests
that the business is heavily dependent on one core market.

The transaction status plot shows that completed transactions dominate the dataset, while cancelled
transactions represent a much smaller share. The purchase quarter distribution adds seasonal context
by showing how transaction counts are spread across Q1, Q2, Q3, and Q4. Together, these panels
explain when revenue happens, where it comes from, and how transaction activity is distributed
across the business cycle.
"""
    print_observation("Figure 2 — Transaction and Sales Story", obs2)
    return ("Figure 2 — Transaction and Sales Story", obs2)


# =========================================================
# Combined Narrative
# =========================================================

def build_combined_narrative():
    narrative = """
Figure 1 focuses on customer behavior using the RFM framework. The distributions of
Recency, Frequency, and Monetary Value show that customers are not homogeneous.
Most customers purchase infrequently and contribute lower monetary value, while a
smaller group appears more active and valuable.

Figure 2 expands the story from customer behavior to business performance. Monthly
revenue shows how sales change over time, the country comparison reveals which
markets contribute most to revenue, transaction status shows the balance between
completed and cancelled activity, and purchase quarter reveals seasonal structure.

Together, these subplot figures create a complete business narrative. Customer value is
unevenly distributed, sales are concentrated in specific markets and time periods, and
transaction outcomes provide context for interpreting overall revenue patterns. This
supports the project goal of translating raw retail transactions into clear and interpretable
business insights.
"""
    return ("Combined Narrative for Layer 8 — Subplots (Storytelling)", narrative)


def print_combined_narrative(narrative_tuple):
    title, text = narrative_tuple
    print(title.upper())
    print()
    print(text.strip())
    print()


# =========================================================
# Main
# =========================================================

def main():
    print("\nStarting Layer 8 — Subplots (Storytelling)...")

    output_dir = ensure_output_dir()
    df, rfm = load_data()
    tx_cols, rfm_cols = standardize_columns(df, rfm)
    df, tx_cols = create_missing_features(df, tx_cols)

    required_rfm = ["Recency", "Frequency", "MonetaryValue"]
    missing_rfm = [col for col in required_rfm if col not in rfm_cols]
    if missing_rfm:
        raise ValueError(f"Missing required RFM columns in rfm_table.csv: {missing_rfm}")

    observations = []

    obs1 = plot_customer_behavior_story(rfm, rfm_cols, output_dir)
    observations.append(obs1)

    obs2 = plot_business_story(df, tx_cols, output_dir)
    observations.append(obs2)

    combined_narrative = build_combined_narrative()
    observations.append(combined_narrative)
    print_combined_narrative(combined_narrative)

    save_observations(output_dir, observations)

    print("Layer 8 completed successfully.")
    print(f"Figures saved in: {output_dir}")


if __name__ == "__main__":
    main()