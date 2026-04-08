import pandas as pd
import numpy as np

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_raw_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Standardize column names
    df.columns = df.columns.str.strip()

    # Fix datatypes
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Keep original dataset naming, then rename in clean_data
    if "Customer ID" in df.columns:
        df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")

    return df


def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    df = df.copy()

    # Rename for consistency
    if "Customer ID" in df.columns:
        df = df.rename(columns={"Customer ID": "CustomerID"})

    # Remove rows where InvoiceDate could not be parsed
    df = df.dropna(subset=["InvoiceDate"])

    # Remove exact duplicates only
    if drop_duplicates:
        df = df.drop_duplicates()

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["LineTotal"] = df["Price"] * df["Quantity"]

    # Business logic:
    # In this dataset, invoices starting with 'C' represent cancelled/credit transactions.
    df["TransactionStatus"] = np.where(
        df["Invoice"].str.startswith("C"),
        "Cancelled",
        "Completed"
    )

    df["PurchaseQuarter"] = df["InvoiceDate"].dt.to_period("Q").astype(str)

    df["PriceCategory"] = pd.cut(
        df["Price"],
        bins=[-np.inf, 1, 5, 20, np.inf],
        labels=["Very Low", "Low", "Medium", "High"]
    )

    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"] = df["InvoiceDate"].dt.day
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()

    return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["CustomerID", "InvoiceDate", "Invoice", "LineTotal", "TransactionStatus"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for compute_rfm: {missing}")

    rfm_df = df.copy()

    # Keep only rows with valid customers
    rfm_df = rfm_df.dropna(subset=["CustomerID"])

    # Use completed transactions only
    rfm_df = rfm_df[rfm_df["TransactionStatus"] == "Completed"].copy()

    if rfm_df.empty:
        raise ValueError("No completed transactions with valid CustomerID found for RFM computation.")

    snapshot_date = rfm_df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        rfm_df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("Invoice", "nunique"),
            MonetaryValue=("LineTotal", "sum")
        )
        .reset_index()
    )

    return rfm


def remove_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    columns: list[str] | None = None,
    z_thresh: float = 3.0
) -> pd.DataFrame:
    out = df.copy()

    if columns is None:
        columns = out.select_dtypes(include=np.number).columns.tolist()

    if not columns:
        raise ValueError("No numeric columns available for outlier removal.")

    if method.lower() == "iqr":
        for col in columns:
            q1 = out[col].quantile(0.25)
            q3 = out[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            out = out[(out[col] >= lower) & (out[col] <= upper)]

    elif method.lower() == "zscore":
        z = np.abs(stats.zscore(out[columns], nan_policy="omit"))
        if len(columns) == 1:
            out = out[z < z_thresh]
        else:
            out = out[(z < z_thresh).all(axis=1)]

    else:
        raise ValueError("method must be either 'iqr' or 'zscore'")

    return out


def normalize(
    df: pd.DataFrame,
    method: str = "log",
    columns: list[str] | None = None
) -> pd.DataFrame:

    out = df.copy()

    if columns is None:
        columns = out.select_dtypes(include=np.number).columns.tolist()

    if not columns:
        raise ValueError("No numeric columns available for normalization.")

    if method.lower() == "log":
        for col in columns:
            min_val = out[col].min()
            if min_val < 0:
                raise ValueError(
                    f"Column '{col}' contains negative values; log1p cannot be safely applied."
                )
            out[col] = np.log1p(out[col])

    elif method.lower() == "standard":
        scaler = StandardScaler()
        out[columns] = scaler.fit_transform(out[columns])

    elif method.lower() == "minmax":
        for col in columns:
            col_min = out[col].min()
            col_max = out[col].max()
            if col_max == col_min:
                out[col] = 0.0
            else:
                out[col] = (out[col] - col_min) / (col_max - col_min)

    else:
        raise ValueError("method must be one of: 'log', 'standard', 'minmax'")

    return out


def run_pca(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    n_components: int = 2,
    standardize: bool = True
) -> tuple[pd.DataFrame, np.ndarray, PCA]:

    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()

    if not columns:
        raise ValueError("No numeric columns available for PCA.")

    X = df[columns].copy()

    if X.isna().any().any():
        raise ValueError("Input to PCA contains NaN values. Clean or impute before PCA.")

    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = X.values

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)

    pca_cols = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(components, columns=pca_cols, index=df.index)

    # Retain CustomerID if present and aligned
    if "CustomerID" in df.columns:
        pca_df["CustomerID"] = df["CustomerID"].values

    return pca_df, pca.explained_variance_ratio_, pca


def compute_correlation(
    df: pd.DataFrame,
    columns: list[str] | None = None
) -> dict[str, pd.DataFrame]:

    if columns is None:
        numeric_df = df.select_dtypes(include=np.number).copy()
    else:
        numeric_df = df[columns].copy()

    if numeric_df.empty:
        raise ValueError("No numeric columns available for correlation analysis.")

    pearson_corr = numeric_df.corr(method="pearson")
    spearman_corr = numeric_df.corr(method="spearman")

    pvals = pd.DataFrame(
        np.ones((numeric_df.shape[1], numeric_df.shape[1])),
        index=numeric_df.columns,
        columns=numeric_df.columns
    )

    for i, col1 in enumerate(numeric_df.columns):
        for j, col2 in enumerate(numeric_df.columns):
            if i == j:
                pvals.loc[col1, col2] = 0.0
            else:
                pair = numeric_df[[col1, col2]].dropna()
                if len(pair) < 2:
                    pvals.loc[col1, col2] = np.nan
                else:
                    _, p = stats.pearsonr(pair[col1], pair[col2])
                    pvals.loc[col1, col2] = p

    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "p_values": pvals
    }


def compute_statistics(
    df: pd.DataFrame,
    columns: list[str] | None = None
) -> pd.DataFrame:

    if columns is None:
        numeric_df = df.select_dtypes(include=np.number).copy()
    else:
        numeric_df = df[columns].copy()

    if numeric_df.empty:
        raise ValueError("No numeric columns available for statistics computation.")

    stats_df = pd.DataFrame({
        "mean": numeric_df.mean(),
        "std": numeric_df.std(),
        "skew": numeric_df.skew(),
        "kurtosis": numeric_df.kurtosis()
    })

    return stats_df


def run_full_pipeline(filepath: str) -> dict[str, pd.DataFrame | np.ndarray | PCA]:

    df_raw = load_raw_data(filepath)
    df_clean = clean_data(df_raw)
    df_features = engineer_features(df_clean)
    print("Duplicate rows in raw:", df_raw.duplicated().sum())
    print("Duplicate rows in clean:", df_clean.duplicated().sum())

    rfm = compute_rfm(df_features)
    rfm_no_outliers = remove_outliers(
        rfm,
        method="iqr",
        columns=["Recency", "Frequency", "MonetaryValue"]
    )
    rfm_log = normalize(
        rfm_no_outliers,
        method="log",
        columns=["Recency", "Frequency", "MonetaryValue"]
    )

    pca_df, explained_var, pca_model = run_pca(
        rfm_log,
        columns=["Recency", "Frequency", "MonetaryValue"],
        n_components=2,
        standardize=True
    )

    corr_results = compute_correlation(
        rfm_no_outliers,
        columns=["Recency", "Frequency", "MonetaryValue"]
    )
    stats_results = compute_statistics(
        rfm_no_outliers,
        columns=["Recency", "Frequency", "MonetaryValue"]
    )

    return {
        "raw": df_raw,
        "clean": df_clean,
        "features": df_features,
        "rfm": rfm,
        "rfm_no_outliers": rfm_no_outliers,
        "rfm_log": rfm_log,
        "pca_df": pca_df,
        "explained_variance_ratio": explained_var,
        "pca_model": pca_model,
        "correlation": corr_results,
        "statistics": stats_results
    }


if __name__ == "__main__":
    FILEPATH = "data/online_retail_II.csv"

    results = run_full_pipeline(FILEPATH)

    print("\nRaw shape:", results["raw"].shape)
    print("Clean shape:", results["clean"].shape)
    print("Feature shape:", results["features"].shape)
    print("RFM shape:", results["rfm"].shape)
    print("RFM without outliers shape:", results["rfm_no_outliers"].shape)

    print("\nRFM head:")
    print(results["rfm"].head())

    print("\nStatistics:")
    print(results["statistics"])

    print("\nPearson correlation:")
    print(results["correlation"]["pearson"])

    print("\nSpearman correlation:")
    print(results["correlation"]["spearman"])

    print("\nPearson p-values:")
    print(results["correlation"]["p_values"])

    print("\nPCA head:")
    print(results["pca_df"].head())

    print("\nExplained variance ratio:")
    print(results["explained_variance_ratio"])
