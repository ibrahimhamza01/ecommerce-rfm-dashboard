import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Input, Output, html
from scipy.stats import gaussian_kde, probplot


RFM_TABLE_FILE = os.path.join("data", "rfm_table.csv")


def load_csv(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    return df


def empty_figure(title: str):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", title=title)
    return fig


def resolve_dataset(current_store, source_mode):
    if source_mode == "rfm":
        return load_csv(RFM_TABLE_FILE)

    if current_store is None or "file_path" not in current_store:
        raise ValueError("Current dataset is not available.")

    return load_csv(current_store["file_path"])


def get_numeric_feature_options(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    options = [{"label": col, "value": col} for col in numeric_cols]

    filtered_cols = [c for c in numeric_cols if c != "CustomerID"]

    x_default = None
    y_default = None
    z_default = None

    preferred = [
        col for col in
        ["Quantity", "Price", "LineTotal", "Recency", "Frequency", "MonetaryValue"]
        if col in filtered_cols
    ]

    base = preferred if preferred else filtered_cols

    if base:
        x_default = base[0]
        y_default = base[1] if len(base) > 1 else base[0]
        z_default = base[2] if len(base) > 2 else base[0]

    return options, x_default, y_default, z_default


def build_histogram(df: pd.DataFrame, feature: str, bins: int):
    values = pd.to_numeric(df[feature], errors="coerce").dropna()
    fig = px.histogram(
        x=values,
        nbins=bins,
        template="plotly_white",
        title=f"Histogram of {feature}",
        labels={"x": feature, "y": "Count"},
    )
    fig.update_layout(showlegend=False)
    return fig


def build_hist_kde(df: pd.DataFrame, feature: str, bins: int):
    values = pd.to_numeric(df[feature], errors="coerce").dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=bins,
            name="Histogram",
            opacity=0.65,
            histnorm="probability density",
        )
    )

    if len(values) >= 5 and values.nunique() > 1:
        kde = gaussian_kde(values)
        xs = np.linspace(values.min(), values.max(), 300)
        ys = kde(xs)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name="KDE",
            )
        )

    fig.update_layout(
        template="plotly_white",
        barmode="overlay",
        title=f"Histogram + KDE of {feature}",
        xaxis_title=feature,
        yaxis_title="Density",
    )
    return fig


def build_boxplot(df: pd.DataFrame, feature: str):
    values = pd.to_numeric(df[feature], errors="coerce").dropna()
    fig = go.Figure()
    fig.add_trace(go.Box(y=values, name=feature, boxmean=True))
    fig.update_layout(template="plotly_white", title=f"Box Plot of {feature}", yaxis_title=feature)
    return fig


def build_violin(df: pd.DataFrame, feature: str):
    values = pd.to_numeric(df[feature], errors="coerce").dropna()
    fig = go.Figure()
    fig.add_trace(go.Violin(y=values, name=feature, box_visible=True, meanline_visible=True))
    fig.update_layout(template="plotly_white", title=f"Violin Plot of {feature}", yaxis_title=feature)
    return fig


def build_line(df: pd.DataFrame, feature: str):
    values = pd.to_numeric(df[feature], errors="coerce").dropna().reset_index(drop=True)
    if len(values) > 3000:
        values = values.sample(3000, random_state=42).sort_index().reset_index(drop=True)

    fig = px.line(
        x=np.arange(len(values)),
        y=values,
        template="plotly_white",
        title=f"Line Plot of {feature}",
        labels={"x": "Observation Index", "y": feature},
    )
    return fig


def build_area(df: pd.DataFrame, feature: str):
    values = pd.to_numeric(df[feature], errors="coerce").dropna().reset_index(drop=True)
    if len(values) > 3000:
        values = values.sample(3000, random_state=42).sort_index().reset_index(drop=True)

    fig = px.area(
        x=np.arange(len(values)),
        y=values,
        template="plotly_white",
        title=f"Area Plot of {feature}",
        labels={"x": "Observation Index", "y": feature},
    )
    return fig


def prepare_xy(df: pd.DataFrame, x_feature: str, y_feature: str, max_rows=10000):
    working = df[[x_feature, y_feature]].copy()
    working[x_feature] = pd.to_numeric(working[x_feature], errors="coerce")
    working[y_feature] = pd.to_numeric(working[y_feature], errors="coerce")
    working = working.dropna()

    if len(working) > max_rows:
        working = working.sample(max_rows, random_state=42)

    return working


def build_scatter(df: pd.DataFrame, x_feature: str, y_feature: str):
    working = prepare_xy(df, x_feature, y_feature)
    fig = px.scatter(
        working,
        x=x_feature,
        y=y_feature,
        template="plotly_white",
        title=f"{y_feature} vs {x_feature}",
        opacity=0.6,
    )
    return fig


def build_regression(df: pd.DataFrame, x_feature: str, y_feature: str):
    working = prepare_xy(df, x_feature, y_feature)
    fig = px.scatter(
        working,
        x=x_feature,
        y=y_feature,
        trendline="ols",
        template="plotly_white",
        title=f"Regression Plot: {y_feature} vs {x_feature}",
        opacity=0.6,
    )
    return fig


def build_hexbin(df: pd.DataFrame, x_feature: str, y_feature: str, bins: int):
    working = prepare_xy(df, x_feature, y_feature)
    fig = px.density_heatmap(
        working,
        x=x_feature,
        y=y_feature,
        nbinsx=bins,
        nbinsy=bins,
        template="plotly_white",
        title=f"Density Heatmap: {y_feature} vs {x_feature}",
    )
    return fig


def build_contour(df: pd.DataFrame, x_feature: str, y_feature: str, bins: int):
    working = prepare_xy(df, x_feature, y_feature)
    fig = px.density_contour(
        working,
        x=x_feature,
        y=y_feature,
        nbinsx=bins,
        nbinsy=bins,
        template="plotly_white",
        title=f"Contour Plot: {y_feature} vs {x_feature}",
    )
    return fig


def build_scatter3d(df: pd.DataFrame, x_feature: str, y_feature: str, z_feature: str):
    working = df[[x_feature, y_feature, z_feature]].copy()
    for c in [x_feature, y_feature, z_feature]:
        working[c] = pd.to_numeric(working[c], errors="coerce")
    working = working.dropna()

    if len(working) > 5000:
        working = working.sample(5000, random_state=42)

    fig = px.scatter_3d(
        working,
        x=x_feature,
        y=y_feature,
        z=z_feature,
        color=z_feature,
        color_continuous_scale="Viridis",
        template="plotly_white",
        title=f"3D Scatter: {z_feature} vs {x_feature}, {y_feature}",
    )
    fig.update_traces(marker=dict(size=3, opacity=0.7))
    return fig


def build_qq(df: pd.DataFrame, feature: str):
    values = pd.to_numeric(df[feature], errors="coerce").dropna()

    if len(values) < 3:
        return empty_figure("Q-Q Plot")

    if len(values) > 3000:
        values = values.sample(3000, random_state=42)

    osm, osr = probplot(values, dist="norm", fit=False)
    slope, intercept = np.polyfit(osm, osr, 1)
    line_x = np.array(osm)
    line_y = slope * line_x + intercept

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Observed Quantiles"))
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", name="Reference Line"))
    fig.update_layout(
        template="plotly_white",
        title=f"Q-Q Plot of {feature}",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
    )
    return fig


def build_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if "CustomerID" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["CustomerID"])

    if numeric_df.shape[1] < 2:
        return empty_figure("Correlation Heatmap")

    corr = numeric_df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(template="plotly_white", title="Correlation Heatmap")
    return fig


def build_summary(df: pd.DataFrame, plot_type: str, x_feature: str, y_feature: str, z_feature: str):
    if plot_type == "heatmap":
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if "CustomerID" in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=["CustomerID"])

        return html.Div(
            [
                html.P(f"Rows: {len(df):,}"),
                html.P(f"Numeric columns used: {numeric_df.shape[1]}"),
                html.P("This heatmap shows pairwise Pearson correlations among numeric variables."),
            ]
        )

    if not x_feature or x_feature not in df.columns:
        return html.P("Selected feature not found.")

    x_vals = pd.to_numeric(df[x_feature], errors="coerce").dropna()

    children = [
        html.P(f"Rows: {len(df):,}"),
        html.P(f"X Feature: {x_feature}"),
        html.P(f"Mean: {x_vals.mean():.2f}" if len(x_vals) else "Mean: N/A"),
        html.P(f"Median: {x_vals.median():.2f}" if len(x_vals) else "Median: N/A"),
        html.P(f"Std Dev: {x_vals.std():.2f}" if len(x_vals) else "Std Dev: N/A"),
        html.P(f"Min: {x_vals.min():.2f}" if len(x_vals) else "Min: N/A"),
        html.P(f"Max: {x_vals.max():.2f}" if len(x_vals) else "Max: N/A"),
    ]

    if plot_type in ["scatter", "regression", "hexbin", "contour"] and y_feature in df.columns:
        pair_df = prepare_xy(df, x_feature, y_feature, max_rows=20000)
        corr = pair_df[x_feature].corr(pair_df[y_feature]) if len(pair_df) > 1 else np.nan
        children.extend(
            [
                html.Hr(),
                html.P(f"Y Feature: {y_feature}"),
                html.P(f"Correlation ({x_feature}, {y_feature}): {corr:.2f}" if pd.notnull(corr) else "Correlation: N/A"),
            ]
        )

    if plot_type == "scatter3d" and z_feature in df.columns:
        children.extend(
            [
                html.Hr(),
                html.P(f"Y Feature: {y_feature}"),
                html.P(f"Z Feature: {z_feature}"),
            ]
        )

    return html.Div(children)


def register_callbacks(app):
    @app.callback(
        [
            Output("numerical-x-feature", "options"),
            Output("numerical-x-feature", "value"),
            Output("numerical-y-feature", "options"),
            Output("numerical-y-feature", "value"),
            Output("numerical-z-feature", "options"),
            Output("numerical-z-feature", "value"),
        ],
        [
            Input("current-data-store", "data"),
            Input("numerical-data-source", "value"),
        ],
        prevent_initial_call=False,
    )
    def populate_numerical_features(current_store, source_mode):
        try:
            df = resolve_dataset(current_store, source_mode)
            options, x_default, y_default, z_default = get_numeric_feature_options(df)
            return options, x_default, options, y_default, options, z_default
        except Exception:
            return [], None, [], None, [], None

    @app.callback(
        [
            Output("numerical-alert", "children"),
            Output("numerical-alert", "color"),
            Output("numerical-summary-text", "children"),
            Output("numerical-main-graph", "figure"),
        ],
        [
            Input("numerical-data-source", "value"),
            Input("numerical-plot-type", "value"),
            Input("numerical-x-feature", "value"),
            Input("numerical-y-feature", "value"),
            Input("numerical-z-feature", "value"),
            Input("numerical-bins-slider", "value"),
            Input("current-data-store", "data"),
        ],
        prevent_initial_call=False,
    )
    def update_numerical_plot(source_mode, plot_type, x_feature, y_feature, z_feature, bins, current_store):
        try:
            df = resolve_dataset(current_store, source_mode)

            if plot_type == "heatmap":
                fig = build_heatmap(df)
            elif plot_type == "histogram":
                fig = build_histogram(df, x_feature, bins)
            elif plot_type == "hist_kde":
                fig = build_hist_kde(df, x_feature, bins)
            elif plot_type == "box":
                fig = build_boxplot(df, x_feature)
            elif plot_type == "violin":
                fig = build_violin(df, x_feature)
            elif plot_type == "line":
                fig = build_line(df, x_feature)
            elif plot_type == "area":
                fig = build_area(df, x_feature)
            elif plot_type == "scatter":
                if not y_feature:
                    return (
                        "Please select a Y feature for scatter plot.",
                        "warning",
                        html.P("No Y feature selected."),
                        empty_figure("Scatter Plot"),
                    )
                fig = build_scatter(df, x_feature, y_feature)
            elif plot_type == "regression":
                if not y_feature:
                    return (
                        "Please select a Y feature for regression plot.",
                        "warning",
                        html.P("No Y feature selected."),
                        empty_figure("Regression Plot"),
                    )
                fig = build_regression(df, x_feature, y_feature)
            elif plot_type == "hexbin":
                if not y_feature:
                    return (
                        "Please select a Y feature for density heatmap.",
                        "warning",
                        html.P("No Y feature selected."),
                        empty_figure("Density Heatmap"),
                    )
                fig = build_hexbin(df, x_feature, y_feature, bins)
            elif plot_type == "contour":
                if not y_feature:
                    return (
                        "Please select a Y feature for contour plot.",
                        "warning",
                        html.P("No Y feature selected."),
                        empty_figure("Contour Plot"),
                    )
                fig = build_contour(df, x_feature, y_feature, bins)
            elif plot_type == "scatter3d":
                if not y_feature or not z_feature:
                    return (
                        "Please select X, Y, and Z features for 3D scatter plot.",
                        "warning",
                        html.P("Missing feature selection for 3D plot."),
                        empty_figure("3D Scatter Plot"),
                    )
                fig = build_scatter3d(df, x_feature, y_feature, z_feature)
            elif plot_type == "qq":
                fig = build_qq(df, x_feature)
            else:
                fig = empty_figure("Numerical Plot")

            summary = build_summary(df, plot_type, x_feature, y_feature, z_feature)

            return (
                f"Displaying {plot_type.replace('_', ' ').title()} for {'RFM Dataset' if source_mode == 'rfm' else 'Current Dataset'}.",
                "success",
                summary,
                fig,
            )

        except Exception as e:
            return (
                f"Failed to build numerical plot: {str(e)}",
                "danger",
                html.P(str(e)),
                empty_figure("Numerical Plot"),
            )