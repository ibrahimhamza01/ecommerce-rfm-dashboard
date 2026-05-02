import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Input, Output, html
from pandas.api.types import (
    is_object_dtype,
    is_categorical_dtype,
    is_bool_dtype,
    is_string_dtype,
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
        df["InvoiceYear"] = df["InvoiceDate"].dt.year.astype("Int64").astype(str)
        df["InvoiceMonth"] = df["InvoiceDate"].dt.month_name().astype(str)
        df["InvoiceQuarter"] = df["InvoiceDate"].dt.to_period("Q").astype(str)

    return df


def get_feature_options(df: pd.DataFrame):
    categorical_cols = []
    numeric_cols = []

    for col in df.columns:
        series = df[col]

        if (
            is_object_dtype(series)
            or is_string_dtype(series)
            or is_categorical_dtype(series)
            or is_bool_dtype(series)
        ):
            categorical_cols.append(col)
        elif is_numeric_dtype(series):
            numeric_cols.append(col)

    # Exclude ID-like numeric fields from value feature defaults where possible
    numeric_value_cols = [c for c in numeric_cols if c not in ["CustomerID", "Customer ID"]]

    cat_options = [{"label": col, "value": col} for col in categorical_cols]
    num_options = [{"label": col, "value": col} for col in numeric_value_cols]

    x_default = categorical_cols[0] if categorical_cols else None
    y_default = categorical_cols[1] if len(categorical_cols) > 1 else x_default
    value_default = numeric_value_cols[0] if numeric_value_cols else None

    return cat_options, x_default, cat_options, y_default, num_options, value_default


def empty_figure(title: str):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", title=title)
    return fig


def top_n_filter(series, top_n):
    top_values = series.astype(str).value_counts().head(top_n).index.tolist()
    return top_values


def build_count_plot(df, x_feature, top_n, normalize, color_hex):
    working = df.copy()
    working[x_feature] = working[x_feature].astype(str)
    top_vals = top_n_filter(working[x_feature], top_n)
    working = working[working[x_feature].isin(top_vals)]

    counts = working[x_feature].value_counts(normalize=normalize).reset_index()
    counts.columns = [x_feature, "Value"]

    fig = px.bar(
        counts,
        x=x_feature,
        y="Value",
        text="Value",
        template="plotly_white",
        title=f"Count Plot of {x_feature}",
    )
    fig.update_traces(marker_color=color_hex, texttemplate="%{text:.2f}", textposition="outside")
    return fig, working


def build_bar_plot(df, x_feature, value_feature, top_n, color_hex):
    working = df[[x_feature, value_feature]].copy()
    working[x_feature] = working[x_feature].astype(str)
    working[value_feature] = pd.to_numeric(working[value_feature], errors="coerce")
    working = working.dropna()

    top_vals = top_n_filter(working[x_feature], top_n)
    working = working[working[x_feature].isin(top_vals)]

    grouped = working.groupby(x_feature, as_index=False)[value_feature].mean()

    fig = px.bar(
        grouped,
        x=x_feature,
        y=value_feature,
        text=value_feature,
        template="plotly_white",
        title=f"Average {value_feature} by {x_feature}",
    )
    fig.update_traces(marker_color=color_hex, texttemplate="%{text:.2f}", textposition="outside")
    return fig, working


def build_grouped_or_stacked_bar(df, x_feature, y_feature, top_n, normalize, color_mode):
    working = df[[x_feature, y_feature]].copy()
    working[x_feature] = working[x_feature].astype(str)
    working[y_feature] = working[y_feature].astype(str)
    working = working.dropna()

    top_vals = top_n_filter(working[x_feature], top_n)
    working = working[working[x_feature].isin(top_vals)]

    counts = working.groupby([x_feature, y_feature]).size().reset_index(name="Count")

    if normalize:
        totals = counts.groupby(x_feature)["Count"].transform("sum")
        counts["Count"] = counts["Count"] / totals

    fig = px.bar(
        counts,
        x=x_feature,
        y="Count",
        color=y_feature,
        barmode=color_mode,
        template="plotly_white",
        title=f"{'Grouped' if color_mode == 'group' else 'Stacked'} Bar Plot: {x_feature} by {y_feature}",
    )
    return fig, working


def build_pie(df, x_feature, top_n, color_hex):
    working = df.copy()
    working[x_feature] = working[x_feature].astype(str)
    top_vals = top_n_filter(working[x_feature], top_n)
    working = working[working[x_feature].isin(top_vals)]

    counts = working[x_feature].value_counts().reset_index()
    counts.columns = [x_feature, "Count"]

    fig = px.pie(
        counts,
        names=x_feature,
        values="Count",
        template="plotly_white",
        title=f"Pie Chart of {x_feature}",
    )
    return fig, working


def build_heatmap(df, x_feature, y_feature, top_n):
    working = df[[x_feature, y_feature]].copy()
    working[x_feature] = working[x_feature].astype(str)
    working[y_feature] = working[y_feature].astype(str)
    working = working.dropna()

    top_vals_x = top_n_filter(working[x_feature], top_n)
    top_vals_y = top_n_filter(working[y_feature], top_n)

    working = working[
        working[x_feature].isin(top_vals_x) & working[y_feature].isin(top_vals_y)
    ]

    matrix = pd.crosstab(working[y_feature], working[x_feature])

    fig = px.imshow(
        matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        template="plotly_white",
        title=f"Heatmap of {x_feature} vs {y_feature}",
        xaxis_title=x_feature,
        yaxis_title=y_feature,
    )
    return fig, working


def build_strip_or_swarm(df, x_feature, y_feature, top_n, color_hex, swarm=False):
    working = df[[x_feature, y_feature]].copy()
    working[x_feature] = working[x_feature].astype(str)
    working[y_feature] = pd.to_numeric(working[y_feature], errors="coerce")
    working = working.dropna()

    top_vals = top_n_filter(working[x_feature], top_n)
    working = working[working[x_feature].isin(top_vals)]

    if len(working) > 5000:
        working = working.sample(5000, random_state=42)

    jitter_val = 0.25 if swarm else 0.45

    fig = px.strip(
        working,
        x=x_feature,
        y=y_feature,
        template="plotly_white",
        title=f"{'Swarm-like' if swarm else 'Strip'} Plot of {y_feature} by {x_feature}",
    )
    fig.update_traces(marker=dict(color=color_hex, opacity=0.65), jitter=jitter_val)
    return fig, working


def build_summary(df, working, x_feature, y_feature):
    missing_pct = 0.0
    if x_feature in df.columns:
        missing_pct = df[x_feature].isna().mean() * 100

    unique_count = working[x_feature].astype(str).nunique() if x_feature in working.columns else 0

    children = [
        html.P(f"Rows in dataset: {len(df):,}"),
        html.P(f"Rows in current plot: {len(working):,}"),
        html.P(f"Primary category: {x_feature}" if x_feature else "Primary category: N/A"),
        html.P(f"Secondary category: {y_feature}" if y_feature else "Secondary category: N/A"),
        html.P(f"Unique categories shown: {unique_count:,}"),
        html.P(f"Missing % in primary category: {missing_pct:.2f}%"),
    ]

    return html.Div(children), len(working), unique_count, missing_pct


def register_callbacks(app):
    @app.callback(
        [
            Output("categorical-x-feature", "options"),
            Output("categorical-x-feature", "value"),
            Output("categorical-y-feature", "options"),
            Output("categorical-y-feature", "value"),
            Output("categorical-value-feature", "options"),
            Output("categorical-value-feature", "value"),
        ],
        [
            Input("current-data-store", "data"),
            Input("categorical-data-source", "value"),
        ],
        prevent_initial_call=False,
    )
    def populate_categorical_features(current_store, source_mode):
        try:
            df = resolve_dataset(current_store, source_mode)
            return get_feature_options(df)
        except Exception:
            return [], None, [], None, [], None

    @app.callback(
        [
            Output("categorical-alert", "children"),
            Output("categorical-alert", "color"),
            Output("categorical-summary-text", "children"),
            Output("categorical-main-graph", "figure"),
            Output("categorical-rows-led", "value"),
            Output("categorical-unique-gauge", "value"),
            Output("categorical-missing-bar", "value"),
        ],
        [
            Input("categorical-data-source", "value"),
            Input("categorical-plot-type", "value"),
            Input("categorical-x-feature", "value"),
            Input("categorical-y-feature", "value"),
            Input("categorical-value-feature", "value"),
            Input("categorical-normalize-switch", "on"),
            Input("categorical-topn-knob", "value"),
            Input("categorical-color-picker", "value"),
            Input("current-data-store", "data"),
        ],
        prevent_initial_call=False,
    )
    def update_categorical_plot(
        source_mode,
        plot_type,
        x_feature,
        y_feature,
        value_feature,
        normalize,
        top_n,
        color_value,
        current_store,
    ):
        try:
            df = resolve_dataset(current_store, source_mode)
            top_n = int(round(top_n)) if top_n is not None else 10
            top_n = max(3, min(top_n, 20))

            if not x_feature:
                return (
                    "Please select a primary categorical feature.",
                    "warning",
                    html.P("No primary category selected."),
                    empty_figure("Categorical Plot"),
                    "0",
                    0,
                    0,
                )

            color_hex = "#1f77b4"
            if isinstance(color_value, dict) and "hex" in color_value:
                color_hex = color_value["hex"]

            if plot_type == "count":
                fig, working = build_count_plot(df, x_feature, top_n, normalize, color_hex)

            elif plot_type == "bar":
                if not value_feature:
                    return (
                        "Please select a numeric feature for bar plot.",
                        "warning",
                        html.P("No numeric feature selected."),
                        empty_figure("Bar Plot"),
                        "0",
                        0,
                        0,
                    )
                fig, working = build_bar_plot(df, x_feature, value_feature, top_n, color_hex)

            elif plot_type == "grouped_bar":
                if not y_feature:
                    return (
                        "Please select a secondary categorical feature for grouped bar plot.",
                        "warning",
                        html.P("No secondary category selected."),
                        empty_figure("Grouped Bar Plot"),
                        "0",
                        0,
                        0,
                    )
                fig, working = build_grouped_or_stacked_bar(df, x_feature, y_feature, top_n, normalize, "group")

            elif plot_type == "stacked_bar":
                if not y_feature:
                    return (
                        "Please select a secondary categorical feature for stacked bar plot.",
                        "warning",
                        html.P("No secondary category selected."),
                        empty_figure("Stacked Bar Plot"),
                        "0",
                        0,
                        0,
                    )
                fig, working = build_grouped_or_stacked_bar(df, x_feature, y_feature, top_n, normalize, "stack")

            elif plot_type == "pie":
                fig, working = build_pie(df, x_feature, top_n, color_hex)

            elif plot_type == "heatmap":
                if not y_feature:
                    return (
                        "Please select a secondary categorical feature for heatmap.",
                        "warning",
                        html.P("No secondary category selected."),
                        empty_figure("Heatmap"),
                        "0",
                        0,
                        0,
                    )
                fig, working = build_heatmap(df, x_feature, y_feature, top_n)

            elif plot_type == "strip":
                if not value_feature:
                    return (
                        "Please select a numeric feature for strip plot.",
                        "warning",
                        html.P("No numeric feature selected."),
                        empty_figure("Strip Plot"),
                        "0",
                        0,
                        0,
                    )
                fig, working = build_strip_or_swarm(df, x_feature, value_feature, top_n, color_hex, swarm=False)

            elif plot_type == "swarm":
                if not value_feature:
                    return (
                        "Please select a numeric feature for swarm-like plot.",
                        "warning",
                        html.P("No numeric feature selected."),
                        empty_figure("Swarm-like Plot"),
                        "0",
                        0,
                        0,
                    )
                fig, working = build_strip_or_swarm(df, x_feature, value_feature, top_n, color_hex, swarm=True)

            else:
                fig = empty_figure("Categorical Plot")
                working = df.copy()

            summary, rows_used, unique_count, missing_pct = build_summary(df, working, x_feature, y_feature)

            return (
                f"Displaying {plot_type.replace('_', ' ').title()} for {'RFM Dataset' if source_mode == 'rfm' else 'Current Dataset'}.",
                "success",
                summary,
                fig,
                f"{rows_used}",
                min(unique_count, 50),
                round(missing_pct, 2),
            )

        except Exception as e:
            return (
                f"Failed to build categorical plot: {str(e)}",
                "danger",
                html.P(str(e)),
                empty_figure("Categorical Plot"),
                "0",
                0,
                0,
            )