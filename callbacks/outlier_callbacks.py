import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore


CACHE_DIR = "cache"
CURRENT_OUTLIER_FILE = os.path.join(CACHE_DIR, "current_outlier_dataset.csv")


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
    default_values = preferred if preferred else numeric_cols[:2]
    default_distribution = default_values[0] if default_values else None

    return options, default_values, default_distribution


def get_outlier_baseline_file(current_store, metadata):
    """
    Always use the Data Cleaning output as the baseline for outlier detection.
    Never reuse the previously outlier-treated dataset as the baseline.
    """
    if metadata and metadata.get("current_file_path"):
        return metadata["current_file_path"]

    if current_store and current_store.get("file_path"):
        return current_store["file_path"]

    return None


def apply_iqr_method(df: pd.DataFrame, features: list, threshold: float):
    result_df = df.copy()
    removed_counts = {}
    combined_mask = pd.Series(False, index=result_df.index)

    for feature in features:
        values = pd.to_numeric(result_df[feature], errors="coerce")
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr

        feature_mask = (values < lower) | (values > upper)
        feature_mask = feature_mask.fillna(False)

        removed_counts[feature] = int(feature_mask.sum())
        combined_mask = combined_mask | feature_mask

    filtered_df = result_df.loc[~combined_mask].copy()
    return filtered_df, removed_counts, int(combined_mask.sum())


def apply_zscore_method(df: pd.DataFrame, features: list, threshold: float):
    result_df = df.copy()
    removed_counts = {}
    combined_mask = pd.Series(False, index=result_df.index)

    for feature in features:
        values = pd.to_numeric(result_df[feature], errors="coerce")
        valid_mask = values.notna()

        feature_mask = pd.Series(False, index=result_df.index)
        if valid_mask.sum() > 1:
            z_vals = np.abs(zscore(values[valid_mask]))
            feature_mask.loc[valid_mask] = z_vals > threshold

        removed_counts[feature] = int(feature_mask.sum())
        combined_mask = combined_mask | feature_mask

    filtered_df = result_df.loc[~combined_mask].copy()
    return filtered_df, removed_counts, int(combined_mask.sum())


def apply_isolation_forest_method(df: pd.DataFrame, features: list, contamination: float):
    result_df = df.copy()
    feature_df = result_df[features].apply(pd.to_numeric, errors="coerce")

    valid_rows = feature_df.dropna().index
    removed_counts = {feature: 0 for feature in features}

    if len(valid_rows) < 10:
        return result_df.copy(), removed_counts, 0

    X = feature_df.loc[valid_rows]

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
    )
    preds = model.fit_predict(X)

    anomaly_mask_valid = preds == -1
    anomaly_indices = X.index[anomaly_mask_valid]

    combined_mask = pd.Series(False, index=result_df.index)
    combined_mask.loc[anomaly_indices] = True

    for feature in features:
        removed_counts[feature] = int(combined_mask.sum())

    filtered_df = result_df.loc[~combined_mask].copy()
    return filtered_df, removed_counts, int(combined_mask.sum())


def build_boxplot_figure(before_df: pd.DataFrame, after_df: pd.DataFrame, features: list):
    fig = go.Figure()

    for feature in features:
        if feature in before_df.columns:
            fig.add_trace(
                go.Box(
                    y=pd.to_numeric(before_df[feature], errors="coerce"),
                    name=f"Before - {feature}",
                    boxmean=True,
                )
            )

        if feature in after_df.columns:
            fig.add_trace(
                go.Box(
                    y=pd.to_numeric(after_df[feature], errors="coerce"),
                    name=f"After - {feature}",
                    boxmean=True,
                )
            )

    fig.update_layout(
        template="plotly_white",
        title="Before vs After Outlier Treatment",
        yaxis_title="Value",
    )
    return fig


def build_outlier_count_figure(removed_counts: dict):
    if not removed_counts:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="No outlier counts available")
        return fig

    df_counts = pd.DataFrame({
        "Feature": list(removed_counts.keys()),
        "RemovedRows": list(removed_counts.values()),
    })

    fig = px.bar(
        df_counts,
        x="Feature",
        y="RemovedRows",
        text="RemovedRows",
        template="plotly_white",
        title="Rows Flagged as Outliers by Feature",
    )
    fig.update_traces(textposition="outside")
    return fig


def build_distribution_figure(before_df: pd.DataFrame, after_df: pd.DataFrame, feature: str):
    fig = go.Figure()

    if not feature or feature not in before_df.columns or feature not in after_df.columns:
        fig.update_layout(template="plotly_white", title="No distribution feature selected")
        return fig

    before_vals = pd.to_numeric(before_df[feature], errors="coerce").dropna()
    after_vals = pd.to_numeric(after_df[feature], errors="coerce").dropna()

    fig.add_trace(
        go.Histogram(
            x=before_vals,
            name=f"Before - {feature}",
            opacity=0.6,
            nbinsx=50,
        )
    )

    fig.add_trace(
        go.Histogram(
            x=after_vals,
            name=f"After - {feature}",
            opacity=0.6,
            nbinsx=50,
        )
    )

    fig.update_layout(
        template="plotly_white",
        barmode="overlay",
        title=f"Distribution Comparison for {feature}",
        xaxis_title=feature,
        yaxis_title="Count",
    )
    return fig


def empty_figure(title: str):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", title=title)
    return fig


def get_threshold_config(method: str):
    if method == "iqr":
        return {
            "label": "Sensitivity Threshold (IQR Multiplier)",
            "min": 0.5,
            "max": 5.0,
            "step": 0.1,
            "value": 1.5,
            "marks": {
                0.5: "0.5",
                1.0: "1.0",
                1.5: "1.5",
                2.0: "2.0",
                3.0: "3.0",
                4.0: "4.0",
                5.0: "5.0",
            },
        }
    if method == "zscore":
        return {
            "label": "Sensitivity Threshold (Z-Score Cutoff)",
            "min": 1.0,
            "max": 5.0,
            "step": 0.1,
            "value": 3.0,
            "marks": {
                1.0: "1.0",
                2.0: "2.0",
                3.0: "3.0",
                4.0: "4.0",
                5.0: "5.0",
            },
        }
    return {
        "label": "Sensitivity Threshold (Isolation Forest Contamination)",
        "min": 0.01,
        "max": 0.20,
        "step": 0.01,
        "value": 0.05,
        "marks": {
            0.01: "0.01",
            0.05: "0.05",
            0.10: "0.10",
            0.15: "0.15",
            0.20: "0.20",
        },
    }


def register_callbacks(app):
    @app.callback(
        [
            Output("outlier-feature-checklist", "options"),
            Output("outlier-feature-checklist", "value"),
            Output("distribution-feature-select", "options"),
            Output("distribution-feature-select", "value"),
        ],
        [
            Input("current-data-store", "data"),
            Input("metadata-store", "data"),
        ],
        prevent_initial_call=False,
    )
    def populate_outlier_features(current_store, metadata):
        baseline_file = get_outlier_baseline_file(current_store, metadata)

        if baseline_file is None:
            return [], [], [], None

        try:
            options, default_values, default_distribution = get_numeric_feature_options(
                baseline_file
            )
            return options, default_values, options, default_distribution
        except Exception:
            return [], [], [], None

    @app.callback(
        [
            Output("distribution-feature-select", "options", allow_duplicate=True),
            Output("distribution-feature-select", "value", allow_duplicate=True),
        ],
        Input("outlier-feature-checklist", "value"),
        prevent_initial_call=True,
    )
    def sync_distribution_feature(selected_features):
        selected_features = selected_features or []
        options = [{"label": feature, "value": feature} for feature in selected_features]
        value = selected_features[0] if selected_features else None
        return options, value

    @app.callback(
        [
            Output("outlier-threshold-label", "children"),
            Output("outlier-threshold-slider", "min"),
            Output("outlier-threshold-slider", "max"),
            Output("outlier-threshold-slider", "step"),
            Output("outlier-threshold-slider", "value"),
            Output("outlier-threshold-slider", "marks"),
        ],
        Input("outlier-method-select", "value"),
        prevent_initial_call=False,
    )
    def update_threshold_controls(method):
        config = get_threshold_config(method)
        return (
            f"{config['label']}: {config['value']:.2f}",
            config["min"],
            config["max"],
            config["step"],
            config["value"],
            config["marks"],
        )

    @app.callback(
        Output("outlier-threshold-label", "children", allow_duplicate=True),
        [
            Input("outlier-threshold-slider", "value"),
            Input("outlier-method-select", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_threshold_label_live(threshold, method):
        config = get_threshold_config(method)
        return f"{config['label']}: {threshold:.2f}"

    @app.callback(
        [
            Output("current-data-store", "data", allow_duplicate=True),
            Output("metadata-store", "data", allow_duplicate=True),
            Output("outlier-alert", "children"),
            Output("outlier-alert", "color"),
            Output("outlier-summary-text", "children"),
            Output("outlier-boxplot-graph", "figure"),
            Output("outlier-count-graph", "figure"),
            Output("outlier-distribution-graph", "figure"),
        ],
        Input("apply-outlier-btn", "n_clicks"),
        [
            State("current-data-store", "data"),
            State("metadata-store", "data"),
            State("outlier-method-select", "value"),
            State("outlier-feature-checklist", "value"),
            State("outlier-threshold-slider", "value"),
            State("distribution-feature-select", "value"),
        ],
        prevent_initial_call=True,
    )
    def apply_outlier_detection(
        n_clicks,
        current_store,
        metadata,
        method,
        selected_features,
        threshold,
        distribution_feature,
    ):
        if not n_clicks:
            raise PreventUpdate

        baseline_file = get_outlier_baseline_file(current_store, metadata)
        if baseline_file is None:
            raise PreventUpdate

        if not selected_features:
            return (
                current_store,
                metadata,
                "Please select at least one numeric feature for outlier detection.",
                "warning",
                html.P("No features selected."),
                empty_figure("Before vs After Boxplots"),
                empty_figure("Outliers Removed by Feature"),
                empty_figure("Distribution Comparison"),
            )

        before_df = load_dataset(baseline_file)

        valid_features = [f for f in selected_features if f in before_df.columns]
        if not valid_features:
            return (
                current_store,
                metadata,
                "Selected features were not found in the active dataset.",
                "danger",
                html.P("No valid features found."),
                empty_figure("Before vs After Boxplots"),
                empty_figure("Outliers Removed by Feature"),
                empty_figure("Distribution Comparison"),
            )

        if not distribution_feature or distribution_feature not in valid_features:
            distribution_feature = valid_features[0]

        if method == "iqr":
            after_df, removed_counts, total_removed = apply_iqr_method(
                before_df, valid_features, threshold
            )
        elif method == "zscore":
            after_df, removed_counts, total_removed = apply_zscore_method(
                before_df, valid_features, threshold
            )
        else:
            after_df, removed_counts, total_removed = apply_isolation_forest_method(
                before_df, valid_features, threshold
            )

        rows_before = len(before_df)
        rows_after = len(after_df)
        retained_pct = (rows_after / rows_before * 100) if rows_before > 0 else 0.0

        save_dataset(after_df, CURRENT_OUTLIER_FILE)

        updated_current_store = {"file_path": CURRENT_OUTLIER_FILE}

        updated_metadata = metadata.copy() if metadata else {}
        # IMPORTANT: keep current_file_path unchanged so re-calculation always uses Data Cleaning baseline
        updated_metadata["outlier_file_path"] = CURRENT_OUTLIER_FILE
        updated_metadata["current_rows"] = int(rows_after)
        updated_metadata["outlier_method"] = method
        updated_metadata["outlier_features"] = valid_features
        updated_metadata["outlier_threshold"] = float(threshold)
        updated_metadata["outlier_applied"] = True

        summary_children = html.Div(
            [
                html.P(f"Rows before outlier detection: {rows_before:,}"),
                html.P(f"Rows after outlier detection: {rows_after:,}"),
                html.P(f"Total rows removed: {total_removed:,}"),
                html.P(f"Total data retained: {retained_pct:.2f}%"),
                html.Hr(),
                html.Ul(
                    [html.Li(f"{feature} outliers flagged: {count:,}") for feature, count in removed_counts.items()]
                ),
            ]
        )

        boxplot_fig = build_boxplot_figure(before_df, after_df, valid_features)
        count_fig = build_outlier_count_figure(removed_counts)
        distribution_fig = build_distribution_figure(before_df, after_df, distribution_feature)

        return (
            updated_current_store,
            updated_metadata,
            "Outlier detection applied successfully. The outlier-treated dataset is now active for the next tabs.",
            "success",
            summary_children,
            boxplot_fig,
            count_fig,
            distribution_fig,
        )

    @app.callback(
        Output("outlier-distribution-graph", "figure", allow_duplicate=True),
        [
            Input("distribution-feature-select", "value"),
            Input("metadata-store", "data"),
            Input("current-data-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_distribution_graph_only(selected_feature, metadata, current_store):
        if not selected_feature:
            return empty_figure("Distribution Comparison")

        baseline_file = get_outlier_baseline_file(current_store, metadata)

        if baseline_file is None:
            return empty_figure("Distribution Comparison")

        if not os.path.exists(baseline_file):
            return empty_figure("Distribution Comparison")

        if not os.path.exists(CURRENT_OUTLIER_FILE):
            return empty_figure("Distribution Comparison")

        before_df = load_dataset(baseline_file)
        after_df = load_dataset(CURRENT_OUTLIER_FILE)

        return build_distribution_figure(before_df, after_df, selected_feature)