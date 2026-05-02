import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
from scipy.stats import boxcox, probplot, normaltest, skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler


CACHE_DIR = "cache"
CURRENT_TRANSFORMED_FILE = os.path.join(CACHE_DIR, "current_transformed_dataset.csv")


def load_dataset(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df


def save_dataset(df: pd.DataFrame, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df_to_save = df.copy()

    if "InvoiceDate" in df_to_save.columns:
        df_to_save["InvoiceDate"] = pd.to_datetime(
            df_to_save["InvoiceDate"], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")

    df_to_save.to_csv(file_path, index=False)


def get_numeric_feature_options(file_path: str):
    df = load_dataset(file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    options = [{"label": col, "value": col} for col in numeric_cols]

    preferred = [col for col in ["Quantity", "Price", "LineTotal"] if col in numeric_cols]
    default_value = preferred[0] if preferred else (numeric_cols[0] if numeric_cols else None)

    return options, default_value


def empty_figure(title: str):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", title=title)
    return fig


def apply_transformation(series: pd.Series, method: str, options: list):
    options = options or []
    s = pd.to_numeric(series, errors="coerce").dropna().copy()

    if s.empty:
        raise ValueError("Selected feature has no valid numeric values.")

    original_index = s.index.copy()
    added_constant = 0.0

    if "handle_zero" in options:
        min_val = s.min()
        if min_val <= 0:
            added_constant = abs(min_val) + 1.0
            s = s + added_constant

    transformed = s.copy()

    if method == "log":
        if (transformed <= 0).any():
            raise ValueError("Log transformation requires positive values. Enable zero/negative handling.")
        transformed = np.log(transformed)

    elif method == "boxcox":
        if (transformed <= 0).any():
            raise ValueError("Box-Cox transformation requires positive values. Enable zero/negative handling.")
        transformed, _ = boxcox(transformed)

    elif method == "standardize":
        scaler = StandardScaler(
            with_mean=("center" in options),
            with_std=("scale" in options or "center" not in options)
        )
        transformed = scaler.fit_transform(transformed.values.reshape(-1, 1)).flatten()

    elif method == "minmax":
        scaler = MinMaxScaler()
        transformed = scaler.fit_transform(transformed.values.reshape(-1, 1)).flatten()

        if "center" in options:
            transformed = transformed - np.mean(transformed)
        if "scale" in options:
            std_val = np.std(transformed)
            if std_val != 0:
                transformed = transformed / std_val

    else:
        raise ValueError("Unsupported transformation method.")

    transformed_series = pd.Series(transformed, index=original_index)

    return transformed_series, added_constant


def build_distribution_figure(before_values: pd.Series, after_values: pd.Series, feature: str, method: str):
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=before_values.dropna(),
            name="Before",
            opacity=0.6,
            nbinsx=40,
        )
    )

    fig.add_trace(
        go.Histogram(
            x=after_values.dropna(),
            name="After",
            opacity=0.6,
            nbinsx=40,
        )
    )

    fig.update_layout(
        template="plotly_white",
        barmode="overlay",
        title=f"Transformed Feature ({feature} - {method})",
        xaxis_title=feature,
        yaxis_title="Count",
    )
    return fig


def build_qq_figure(before_values: pd.Series, after_values: pd.Series, feature: str):
    before_clean = before_values.dropna()
    after_clean = after_values.dropna()

    if len(before_clean) < 3 or len(after_clean) < 3:
        return empty_figure("Normal Q-Q Plot")

    if len(before_clean) > 3000:
        before_clean = before_clean.sample(3000, random_state=42)
    if len(after_clean) > 3000:
        after_clean = after_clean.sample(3000, random_state=42)

    osm_before, osr_before = probplot(before_clean, dist="norm", fit=False)
    slope_b, intercept_b = np.polyfit(osm_before, osr_before, 1)
    line_y_before = slope_b * np.array(osm_before) + intercept_b

    osm_after, osr_after = probplot(after_clean, dist="norm", fit=False)
    slope_a, intercept_a = np.polyfit(osm_after, osr_after, 1)
    line_y_after = slope_a * np.array(osm_after) + intercept_a

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=osm_before,
            y=osr_before,
            mode="markers",
            name="Before Quantiles",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(osm_before),
            y=line_y_before,
            mode="lines",
            name="Before Reference",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=osm_after,
            y=osr_after,
            mode="markers",
            name="After Quantiles",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(osm_after),
            y=line_y_after,
            mode="lines",
            name="After Reference",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=f"Normal Q-Q Plot ({feature})",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
    )
    return fig


def build_boxplot_figure(before_values: pd.Series, after_values: pd.Series, feature: str):
    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=before_values.dropna(),
            name="Before",
            boxmean=True,
        )
    )

    fig.add_trace(
        go.Box(
            y=after_values.dropna(),
            name="After",
            boxmean=True,
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=f"Box Plot (Before vs After) - {feature}",
        yaxis_title=feature,
    )
    return fig


def normality_pvalue(values: pd.Series):
    vals = pd.to_numeric(values, errors="coerce").dropna()

    if len(vals) < 8:
        return None

    if len(vals) > 5000:
        vals = vals.sample(5000, random_state=42)

    _, p_value = normaltest(vals)
    return p_value


def get_latest_transformed_column(df: pd.DataFrame, base_feature: str):
    matching_cols = [col for col in df.columns if col == base_feature or col.startswith(f"{base_feature}_")]

    if not matching_cols:
        return None

    latest_col = sorted(matching_cols, key=lambda x: (x.count("_"), len(x)))[-1]
    return latest_col


def build_transformation_history(input_col: str, output_col: str):
    return input_col.replace("_", " → ") + f" → {output_col.split('_')[-1]}"


def register_callbacks(app):
    @app.callback(
        [
            Output("transform-feature-select", "options"),
            Output("transform-feature-select", "value"),
        ],
        Input("current-data-store", "data"),
        prevent_initial_call=False,
    )
    def populate_transform_features(current_store):
        if current_store is None or "file_path" not in current_store:
            return [], None

        try:
            return get_numeric_feature_options(current_store["file_path"])
        except Exception:
            return [], None

    @app.callback(
        [
            Output("current-data-store", "data", allow_duplicate=True),
            Output("metadata-store", "data", allow_duplicate=True),
            Output("transformation-alert", "children"),
            Output("transformation-alert", "color"),
            Output("transformation-summary-text", "children"),
            Output("transformation-distribution-graph", "figure"),
            Output("transformation-qq-graph", "figure"),
            Output("transformation-boxplot-graph", "figure"),
        ],
        Input("apply-transformation-btn", "n_clicks"),
        [
            State("current-data-store", "data"),
            State("metadata-store", "data"),
            State("transform-feature-select", "value"),
            State("transform-method-select", "value"),
            State("transform-source-mode", "value"),
            State("transform-options-checklist", "value"),
        ],
        prevent_initial_call=True,
    )
    def apply_feature_transformation(
        n_clicks,
        current_store,
        metadata,
        selected_feature,
        transform_method,
        transform_source_mode,
        transform_options,
    ):
        if not n_clicks:
            raise PreventUpdate

        if current_store is None or "file_path" not in current_store:
            raise PreventUpdate

        if not selected_feature:
            return (
                current_store,
                metadata,
                "Please select a feature to transform.",
                "warning",
                html.P("No feature selected."),
                empty_figure("Transformed Feature"),
                empty_figure("Normal Q-Q Plot"),
                empty_figure("Box Plot (Before vs After)"),
            )

        df = load_dataset(current_store["file_path"])

        if selected_feature not in df.columns:
            return (
                current_store,
                metadata,
                "Selected base feature was not found in the active dataset.",
                "danger",
                html.P("Invalid feature."),
                empty_figure("Transformed Feature"),
                empty_figure("Normal Q-Q Plot"),
                empty_figure("Box Plot (Before vs After)"),
            )

        if transform_source_mode == "latest":
            input_col = get_latest_transformed_column(df, selected_feature)
        else:
            input_col = selected_feature

        if input_col is None or input_col not in df.columns:
            return (
                current_store,
                metadata,
                "Could not determine the input column for transformation.",
                "danger",
                html.P("Transformation source resolution failed."),
                empty_figure("Transformed Feature"),
                empty_figure("Normal Q-Q Plot"),
                empty_figure("Box Plot (Before vs After)"),
            )

        before_values = pd.to_numeric(df[input_col], errors="coerce")

        try:
            transformed_series, added_constant = apply_transformation(
                before_values,
                transform_method,
                transform_options,
            )
        except Exception as e:
            return (
                current_store,
                metadata,
                f"Transformation failed: {str(e)}",
                "danger",
                html.P(str(e)),
                empty_figure("Transformed Feature"),
                empty_figure("Normal Q-Q Plot"),
                empty_figure("Box Plot (Before vs After)"),
            )

        df_after = df.copy()
        transformed_col_name = f"{input_col}_{transform_method}"

        if transformed_col_name in df_after.columns:
            suffix = 2
            while f"{transformed_col_name}_{suffix}" in df_after.columns:
                suffix += 1
            transformed_col_name = f"{transformed_col_name}_{suffix}"

        df_after[transformed_col_name] = np.nan
        df_after.loc[transformed_series.index, transformed_col_name] = transformed_series.values

        save_dataset(df_after, CURRENT_TRANSFORMED_FILE)

        updated_current_store = {"file_path": CURRENT_TRANSFORMED_FILE}

        updated_metadata = metadata.copy() if metadata else {}
        updated_metadata["current_file_path"] = CURRENT_TRANSFORMED_FILE
        updated_metadata["transformation_applied"] = True
        updated_metadata["base_feature"] = selected_feature
        updated_metadata["transformation_input_column"] = input_col
        updated_metadata["transformed_feature"] = transformed_col_name
        updated_metadata["transformation_method"] = transform_method
        updated_metadata["transformation_source_mode"] = transform_source_mode
        updated_metadata["transformation_options"] = transform_options or []

        history = updated_metadata.get("transformation_history", [])
        history = history + [
            {
                "base_feature": selected_feature,
                "input_column": input_col,
                "output_column": transformed_col_name,
                "method": transform_method,
                "source_mode": transform_source_mode,
            }
        ]
        updated_metadata["transformation_history"] = history

        after_values = df_after[transformed_col_name]

        before_skew = skew(before_values.dropna()) if len(before_values.dropna()) > 2 else np.nan
        after_skew = skew(after_values.dropna()) if len(after_values.dropna()) > 2 else np.nan

        before_p = normality_pvalue(before_values)
        after_p = normality_pvalue(after_values)

        before_p_text = f"{before_p:.2f}" if before_p is not None else "N/A"
        after_p_text = f"{after_p:.2f}" if after_p is not None else "N/A"

        if after_p is not None and after_p >= 0.05:
            verdict = "Data is closer to normal and more suitable for downstream analysis."
            verdict_color = "success"
        else:
            verdict = "Data still shows evidence against normality. Further transformation may be needed."
            verdict_color = "warning"

        history_text = " → ".join([step["output_column"] for step in history]) if history else "N/A"

        summary_children = html.Div(
            [
                html.P(f"Base feature selected: {selected_feature}"),
                html.P(f"Transformation source mode: {transform_source_mode}"),
                html.P(f"Input column used: {input_col}"),
                html.P(f"Output column created: {transformed_col_name}"),
                html.P(f"Transformation type: {transform_method}"),
                html.P(f"Added constant for zero/negative handling: {added_constant:.2f}"),
                html.Hr(),
                html.P(f"Skewness before: {before_skew:.2f}" if pd.notnull(before_skew) else "Skewness before: N/A"),
                html.P(f"Skewness after: {after_skew:.2f}" if pd.notnull(after_skew) else "Skewness after: N/A"),
                html.P(f"Normality p-value before: {before_p_text}"),
                html.P(f"Normality p-value after: {after_p_text}"),
                html.Hr(),
                html.P([html.Strong("History: "), history_text]),
                html.P([html.Strong("Verdict: "), verdict]),
            ]
        )

        distribution_fig = build_distribution_figure(
            before_values=before_values,
            after_values=after_values,
            feature=input_col,
            method=transform_method,
        )

        qq_fig = build_qq_figure(
            before_values=before_values,
            after_values=after_values,
            feature=input_col,
        )

        boxplot_fig = build_boxplot_figure(
            before_values=before_values,
            after_values=after_values,
            feature=input_col,
        )

        return (
            updated_current_store,
            updated_metadata,
            "Transformation applied successfully. The transformed dataset is now active for the next tabs.",
            verdict_color,
            summary_children,
            distribution_fig,
            qq_fig,
            boxplot_fig,
        )