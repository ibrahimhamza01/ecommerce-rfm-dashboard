import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Input, Output, State, html, dcc
from dash.exceptions import PreventUpdate
from scipy.stats import ttest_1samp, ttest_ind, f_oneway, chi2_contingency

from pandas.api.types import (
    is_object_dtype,
    is_string_dtype,
    is_categorical_dtype,
    is_bool_dtype,
    is_numeric_dtype,
)

RFM_TABLE_FILE = os.path.join("data", "rfm_table.csv")


def load_csv(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    return df


def resolve_dataset(current_store, source_mode):
    if source_mode == "rfm":
        df = load_csv(RFM_TABLE_FILE)

        if "Recency" in df.columns:
            df["RecencySegment"] = pd.cut(
                pd.to_numeric(df["Recency"], errors="coerce"),
                bins=[-np.inf, 30, 90, 180, np.inf],
                labels=["Very Recent", "Recent", "Aging", "Dormant"],
                include_lowest=True,
            ).astype(str)

        if "Frequency" in df.columns:
            df["FrequencySegment"] = pd.cut(
                pd.to_numeric(df["Frequency"], errors="coerce"),
                bins=[-np.inf, 1, 3, 10, np.inf],
                labels=["One-Time", "Occasional", "Frequent", "Loyal"],
                include_lowest=True,
            ).astype(str)

        if "MonetaryValue" in df.columns:
            mv = pd.to_numeric(df["MonetaryValue"], errors="coerce")
            q1 = mv.quantile(0.25)
            q2 = mv.quantile(0.50)
            q3 = mv.quantile(0.75)

            df["MonetarySegment"] = pd.cut(
                mv,
                bins=[-np.inf, q1, q2, q3, np.inf],
                labels=["Low", "Medium", "High", "Premium"],
                include_lowest=True,
            ).astype(str)

        return df

    if current_store is None or "file_path" not in current_store:
        raise ValueError("Current dataset is not available.")

    df = load_csv(current_store["file_path"])

    if "Invoice" in df.columns:
        df["InvoiceType"] = np.where(
            df["Invoice"].astype(str).str.startswith("C", na=False),
            "Cancelled",
            "Completed",
        )

    if "Quantity" in df.columns:
        qty = pd.to_numeric(df["Quantity"], errors="coerce")
        df["QuantityType"] = np.where(qty < 0, "Return", "Order")

    if "Price" in df.columns:
        price = pd.to_numeric(df["Price"], errors="coerce")
        df["PriceBand"] = pd.cut(
            price,
            bins=[-np.inf, 0, 2, 5, 10, np.inf],
            labels=["Negative/Zero", "Low", "Medium", "High", "Premium"],
            include_lowest=True,
        ).astype(str)

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df["InvoiceYear"] = df["InvoiceDate"].dt.year.astype("Int64").astype(str)
        df["InvoiceQuarter"] = df["InvoiceDate"].dt.to_period("Q").astype(str)
        df["InvoiceMonth"] = df["InvoiceDate"].dt.month_name().astype(str)

    return df


def get_column_options(df):
    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        series = df[col]

        if is_numeric_dtype(series):
            numeric_cols.append(col)

        if (
            is_object_dtype(series)
            or is_string_dtype(series)
            or is_categorical_dtype(series)
            or is_bool_dtype(series)
        ):
            categorical_cols.append(col)

    numeric_cols = [c for c in numeric_cols if c not in ["CustomerID", "Customer ID"]]

    preferred_numeric = [
        col for col in
        ["Quantity", "Price", "LineTotal", "Revenue", "Recency", "Frequency", "MonetaryValue"]
        if col in numeric_cols
    ]

    numeric_defaults = preferred_numeric[:4] if preferred_numeric else numeric_cols[:4]

    preferred_categories = [
        col for col in
        [
            "Country",
            "InvoiceType",
            "QuantityType",
            "PriceBand",
            "InvoiceYear",
            "InvoiceQuarter",
            "RecencySegment",
            "FrequencySegment",
            "MonetarySegment",
        ]
        if col in categorical_cols
    ]

    ordered_categories = preferred_categories + [
        c for c in categorical_cols if c not in preferred_categories
    ]

    numeric_options = [{"label": col, "value": col} for col in numeric_cols]
    categorical_options = [{"label": col, "value": col} for col in ordered_categories]

    test_numeric_default = numeric_defaults[0] if numeric_defaults else None
    cat1_default = ordered_categories[0] if ordered_categories else None
    cat2_default = ordered_categories[1] if len(ordered_categories) > 1 else cat1_default

    return (
        numeric_options,
        numeric_defaults,
        numeric_options,
        test_numeric_default,
        categorical_options,
        cat1_default,
        categorical_options,
        cat2_default,
    )


def build_descriptive_stats(df, numeric_cols):
    rows = []

    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        clean = values.dropna()

        rows.append(
            {
                "Column": col,
                "Count": f"{clean.count():,.0f}",
                "Missing %": f"{values.isna().mean() * 100:.2f}",
                "Mean": f"{clean.mean():.2f}" if len(clean) else "N/A",
                "Median": f"{clean.median():.2f}" if len(clean) else "N/A",
                "Std Dev": f"{clean.std():.2f}" if len(clean) else "N/A",
                "Min": f"{clean.min():.2f}" if len(clean) else "N/A",
                "Max": f"{clean.max():.2f}" if len(clean) else "N/A",
                "Skewness": f"{clean.skew():.2f}" if len(clean) > 2 else "N/A",
                "Kurtosis": f"{clean.kurtosis():.2f}" if len(clean) > 3 else "N/A",
            }
        )

    return pd.DataFrame(rows)


def build_correlation_heatmap(df, numeric_cols, method):
    if not numeric_cols or len(numeric_cols) < 2:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="Select at least two numeric columns.")
        return fig

    corr_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce").corr(method=method)

    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(
        template="plotly_white",
        title=f"{method.title()} Correlation Heatmap",
    )
    return fig


def strongest_correlations(df, numeric_cols, method):
    if len(numeric_cols) < 2:
        return "Not enough numeric columns for correlation analysis."

    corr = df[numeric_cols].apply(pd.to_numeric, errors="coerce").corr(method=method)

    pairs = []
    for i, col1 in enumerate(corr.columns):
        for col2 in corr.columns[i + 1:]:
            val = corr.loc[col1, col2]
            if pd.notnull(val):
                pairs.append((col1, col2, val))

    if not pairs:
        return "No valid correlation pairs found."

    strongest_pos = max(pairs, key=lambda x: x[2])
    strongest_neg = min(pairs, key=lambda x: x[2])

    return html.Div(
        [
            html.P(
                f"Strongest positive correlation: {strongest_pos[0]} vs {strongest_pos[1]} = {strongest_pos[2]:.2f}"
            ),
            html.P(
                f"Strongest negative correlation: {strongest_neg[0]} vs {strongest_neg[1]} = {strongest_neg[2]:.2f}"
            ),
        ]
    )


def run_hypothesis_test(df, test_type, numeric_col, cat1, cat2, hypothesized_mean):
    if test_type == "one_sample_t":
        if not numeric_col or numeric_col not in df.columns:
            return html.P("Select a valid numeric column.")

        values = pd.to_numeric(df[numeric_col], errors="coerce").dropna()

        if len(values) < 3:
            return html.P("Not enough observations for one-sample t-test.")

        if len(values) > 10000:
            values = values.sample(10000, random_state=42)

        stat, p = ttest_1samp(values, popmean=hypothesized_mean)

        return html.Div(
            [
                html.H5("One-Sample T-Test"),
                html.P(f"Column tested: {numeric_col}"),
                html.P(f"Hypothesized mean: {hypothesized_mean:.2f}"),
                html.P(f"Test statistic: {stat:.2f}"),
                html.P(f"p-value: {p:.2f}"),
                html.P(
                    "Verdict: Reject the null hypothesis."
                    if p < 0.05 else
                    "Verdict: Fail to reject the null hypothesis."
                ),
            ]
        )

    if test_type in ["two_sample_t", "anova"]:
        if not numeric_col or not cat1:
            return html.P("Select a numeric column and category column.")

        if numeric_col not in df.columns or cat1 not in df.columns:
            return html.P("Selected columns were not found.")

        working = df[[numeric_col, cat1]].copy()
        working[numeric_col] = pd.to_numeric(working[numeric_col], errors="coerce")
        working[cat1] = working[cat1].astype(str)
        working = working.dropna()

        top_groups = working[cat1].value_counts().head(5).index.tolist()
        working = working[working[cat1].isin(top_groups)]

        groups = [
            group[numeric_col].dropna()
            for _, group in working.groupby(cat1)
            if len(group[numeric_col].dropna()) >= 3
        ]

        if test_type == "two_sample_t":
            if len(groups) < 2:
                return html.P("Need at least two valid category groups.")

            g1 = groups[0]
            g2 = groups[1]

            if len(g1) > 5000:
                g1 = g1.sample(5000, random_state=42)
            if len(g2) > 5000:
                g2 = g2.sample(5000, random_state=42)

            stat, p = ttest_ind(g1, g2, equal_var=False)

            return html.Div(
                [
                    html.H5("Two-Sample T-Test"),
                    html.P(f"Numeric column: {numeric_col}"),
                    html.P(f"Grouping column: {cat1}"),
                    html.P(f"Groups compared: {top_groups[0]} vs {top_groups[1]}"),
                    html.P(f"Test statistic: {stat:.2f}"),
                    html.P(f"p-value: {p:.2f}"),
                    html.P(
                        "Verdict: Reject the null hypothesis."
                        if p < 0.05 else
                        "Verdict: Fail to reject the null hypothesis."
                    ),
                ]
            )

        if len(groups) < 2:
            return html.P("Need at least two valid category groups for ANOVA.")

        sampled_groups = []
        for g in groups:
            sampled_groups.append(g.sample(5000, random_state=42) if len(g) > 5000 else g)

        stat, p = f_oneway(*sampled_groups)

        return html.Div(
            [
                html.H5("ANOVA Test"),
                html.P(f"Numeric column: {numeric_col}"),
                html.P(f"Grouping column: {cat1}"),
                html.P(f"Groups used: {', '.join(top_groups)}"),
                html.P(f"F-statistic: {stat:.2f}"),
                html.P(f"p-value: {p:.2f}"),
                html.P(
                    "Verdict: Reject the null hypothesis."
                    if p < 0.05 else
                    "Verdict: Fail to reject the null hypothesis."
                ),
            ]
        )

    if test_type == "chi_square":
        if not cat1 or not cat2:
            return html.P("Select two categorical columns.")

        if cat1 not in df.columns or cat2 not in df.columns:
            return html.P("Selected categorical columns were not found.")

        working = df[[cat1, cat2]].copy()
        working[cat1] = working[cat1].astype(str)
        working[cat2] = working[cat2].astype(str)
        working = working.dropna()

        top_cat1 = working[cat1].value_counts().head(10).index
        top_cat2 = working[cat2].value_counts().head(10).index

        working = working[working[cat1].isin(top_cat1) & working[cat2].isin(top_cat2)]

        table = pd.crosstab(working[cat1], working[cat2])

        if table.shape[0] < 2 or table.shape[1] < 2:
            return html.P("Need at least a 2x2 contingency table.")

        stat, p, dof, _ = chi2_contingency(table)

        return html.Div(
            [
                html.H5("Chi-Square Test of Independence"),
                html.P(f"Category 1: {cat1}"),
                html.P(f"Category 2: {cat2}"),
                html.P(f"Chi-square statistic: {stat:.2f}"),
                html.P(f"Degrees of freedom: {dof}"),
                html.P(f"p-value: {p:.2f}"),
                html.P(
                    "Verdict: Reject the null hypothesis. The variables appear associated."
                    if p < 0.05 else
                    "Verdict: Fail to reject the null hypothesis. No strong association detected."
                ),
            ]
        )

    return html.P("Invalid test selected.")


def register_callbacks(app):
    @app.callback(
        [
            Output("stats-numeric-columns", "options"),
            Output("stats-numeric-columns", "value"),
            Output("stats-test-numeric", "options"),
            Output("stats-test-numeric", "value"),
            Output("stats-test-category-1", "options"),
            Output("stats-test-category-1", "value"),
            Output("stats-test-category-2", "options"),
            Output("stats-test-category-2", "value"),
        ],
        [
            Input("current-data-store", "data"),
            Input("stats-data-source", "value"),
        ],
        prevent_initial_call=False,
    )
    def populate_statistics_columns(current_store, source_mode):
        try:
            df = resolve_dataset(current_store, source_mode)
            return get_column_options(df)
        except Exception:
            return [], [], [], None, [], None, [], None

    @app.callback(
        [
            Output("stats-alert", "children"),
            Output("stats-alert", "color"),
            Output("stats-summary-text", "children"),
            Output("stats-descriptive-table", "data"),
            Output("stats-descriptive-table", "columns"),
            Output("stats-correlation-graph", "figure"),
            Output("stats-test-results", "children"),
        ],
        Input("run-stats-btn", "n_clicks"),
        [
            State("current-data-store", "data"),
            State("stats-data-source", "value"),
            State("stats-numeric-columns", "value"),
            State("stats-corr-method", "value"),
            State("stats-test-type", "value"),
            State("stats-test-numeric", "value"),
            State("stats-test-category-1", "value"),
            State("stats-test-category-2", "value"),
            State("stats-hypothesized-mean", "value"),
        ],
        prevent_initial_call=True,
    )
    def run_statistics(
        n_clicks,
        current_store,
        source_mode,
        numeric_cols,
        corr_method,
        test_type,
        test_numeric,
        cat1,
        cat2,
        hypothesized_mean,
    ):
        if not n_clicks:
            raise PreventUpdate

        try:
            df = resolve_dataset(current_store, source_mode)
            numeric_cols = numeric_cols or []

            desc_df = build_descriptive_stats(df, numeric_cols)
            table_data = desc_df.to_dict("records")
            table_columns = [{"name": col, "id": col} for col in desc_df.columns]

            corr_fig = build_correlation_heatmap(df, numeric_cols, corr_method)

            summary = html.Div(
                [
                    html.P(f"Data source: {'RFM Dataset' if source_mode == 'rfm' else 'Current Dataset'}"),
                    html.P(f"Rows: {len(df):,}"),
                    html.P(f"Columns: {df.shape[1]:,}"),
                    html.P(f"Selected numeric columns: {len(numeric_cols)}"),
                    html.Hr(),
                    strongest_correlations(df, numeric_cols, corr_method),
                ]
            )

            test_results = run_hypothesis_test(
                df=df,
                test_type=test_type,
                numeric_col=test_numeric,
                cat1=cat1,
                cat2=cat2,
                hypothesized_mean=float(hypothesized_mean or 0),
            )

            return (
                "Statistics completed successfully.",
                "success",
                summary,
                table_data,
                table_columns,
                corr_fig,
                test_results,
            )

        except Exception as e:
            return (
                f"Statistics failed: {str(e)}",
                "danger",
                html.P(str(e)),
                [],
                [],
                go.Figure().update_layout(template="plotly_white", title="Correlation Heatmap"),
                html.P(str(e)),
            )

    @app.callback(
        Output("download-stats-csv", "data"),
        Input("download-stats-btn", "n_clicks"),
        [
            State("current-data-store", "data"),
            State("stats-data-source", "value"),
            State("stats-numeric-columns", "value"),
        ],
        prevent_initial_call=True,
    )
    def download_statistics(n_clicks, current_store, source_mode, numeric_cols):
        if not n_clicks:
            raise PreventUpdate

        df = resolve_dataset(current_store, source_mode)
        desc_df = build_descriptive_stats(df, numeric_cols or [])

        return dcc.send_data_frame(
            desc_df.to_csv,
            "descriptive_statistics.csv",
            index=False,
        )