import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


RFM_TABLE_FILE = os.path.join("data", "rfm_table.csv")
RFM_PCA_FILE = os.path.join("data", "rfm_pca.csv")


def load_csv(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def empty_figure(title: str):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", title=title)
    return fig


def build_variance_figure(explained_variance_ratio: np.ndarray):
    pcs = [f"PC{i+1}" for i in range(len(explained_variance_ratio))]
    df_var = pd.DataFrame(
        {
            "Principal Component": pcs,
            "Explained Variance Ratio": explained_variance_ratio,
        }
    )

    fig = px.bar(
        df_var,
        x="Principal Component",
        y="Explained Variance Ratio",
        text="Explained Variance Ratio",
        template="plotly_white",
        title="Explained Variance by Principal Component",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_yaxes(tickformat=".0%")
    return fig


def build_pca_scatter_2d(pca_df: pd.DataFrame, color_col: str = "PC1"):
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color=color_col if color_col in pca_df.columns else None,
        color_continuous_scale="Viridis" if color_col in pca_df.columns else None,
        template="plotly_white",
        title="PCA Projection (2D)",
        hover_data=[c for c in ["CustomerID"] if c in pca_df.columns],
    )
    return fig


def build_pca_scatter_3d(pca_df: pd.DataFrame, color_col: str = "PC1"):
    fig = px.scatter_3d(
        pca_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color=color_col if color_col in pca_df.columns else None,
        color_continuous_scale="Viridis" if color_col in pca_df.columns else None,
        template="plotly_white",
        title="PCA Projection (3D)",
        hover_data=[c for c in ["CustomerID"] if c in pca_df.columns],
    )
    fig.update_traces(marker=dict(size=3, opacity=0.7))
    return fig


def build_loadings_figure(loadings_df: pd.DataFrame, show_loadings: bool):
    if not show_loadings:
        return empty_figure("Loadings hidden. Enable 'Show Loadings' to display them.")

    fig = px.imshow(
        loadings_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        origin="lower",
    )
    fig.update_layout(
        template="plotly_white",
        title="PCA Loadings Heatmap",
        xaxis_title="Principal Components",
        yaxis_title="Features",
    )
    return fig


def run_live_rfm_pca(selected_features, n_components, scale_features):
    df = load_csv(RFM_TABLE_FILE)

    valid_features = [col for col in selected_features if col in df.columns]
    if len(valid_features) < 2:
        raise ValueError("Please select at least 2 valid RFM features.")

    working = df[["CustomerID"] + valid_features].copy()
    X = working[valid_features].apply(pd.to_numeric, errors="coerce").dropna()

    if X.shape[0] < 3:
        raise ValueError("Not enough complete rows remain after removing missing values.")

    customer_ids = working.loc[X.index, "CustomerID"].values

    max_components = min(X.shape[0], X.shape[1])
    n_components = min(int(n_components), max_components)

    X_input = X.copy()
    if scale_features:
        scaler = StandardScaler()
        X_input = scaler.fit_transform(X_input)
    else:
        X_input = X_input.values

    pca_model = PCA(n_components=n_components, random_state=42)
    scores = pca_model.fit_transform(X_input)

    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(scores, columns=pc_cols)
    pca_df["CustomerID"] = customer_ids

    loadings = pd.DataFrame(
        pca_model.components_.T,
        index=valid_features,
        columns=pc_cols,
    )

    return {
        "mode_label": "Live PCA on RFM Table",
        "n_observations": len(X),
        "features_used": valid_features,
        "explained_variance_ratio": pca_model.explained_variance_ratio_,
        "pca_df": pca_df,
        "loadings": loadings,
    }


def use_precomputed_rfm_pca(n_components):
    df = load_csv(RFM_PCA_FILE)

    required = {"PC1", "PC2", "PC3", "CustomerID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"rfm_pca.csv is missing required columns: {sorted(missing)}")

    n_components = int(n_components)
    if n_components not in [2, 3]:
        n_components = 3

    pc_cols = ["PC1", "PC2"] if n_components == 2 else ["PC1", "PC2", "PC3"]
    pca_df = df[pc_cols + ["CustomerID"]].copy()

    # Approximate variance from the precomputed PCA coordinates
    variances = pca_df[pc_cols].var(ddof=1).values
    total = variances.sum()
    explained_variance_ratio = variances / total if total > 0 else np.zeros_like(variances)

    return {
        "mode_label": "Precomputed PCA from rfm_pca.csv",
        "n_observations": len(pca_df),
        "features_used": ["Recency", "Frequency", "MonetaryValue"],
        "explained_variance_ratio": explained_variance_ratio,
        "pca_df": pca_df,
        "loadings": None,
    }


def register_callbacks(app):
    @app.callback(
        [
            Output("pca-feature-select", "disabled"),
            Output("pca-feature-select", "value"),
        ],
        Input("pca-source-mode", "value"),
        prevent_initial_call=False,
    )
    def sync_pca_feature_control(source_mode):
        if source_mode == "rfm_pca":
            return True, ["Recency", "Frequency", "MonetaryValue"]
        return False, ["Recency", "Frequency", "MonetaryValue"]

    @app.callback(
        [
            Output("pca-alert", "children"),
            Output("pca-alert", "color"),
            Output("pca-summary-text", "children"),
            Output("pca-variance-graph", "figure"),
            Output("pca-scatter-graph", "figure"),
            Output("pca-loadings-graph", "figure"),
        ],
        Input("run-pca-btn", "n_clicks"),
        [
            State("pca-source-mode", "value"),
            State("pca-components-select", "value"),
            State("pca-feature-select", "value"),
            State("pca-options-checklist", "value"),
        ],
        prevent_initial_call=True,
    )
    def run_pca_analysis(n_clicks, source_mode, n_components, selected_features, pca_options):
        if not n_clicks:
            raise PreventUpdate

        try:
            show_loadings = "show_loadings" in (pca_options or [])
            scale_features = "scale" in (pca_options or [])

            if source_mode == "rfm_pca":
                result = use_precomputed_rfm_pca(n_components)
                loadings_fig = empty_figure("Loadings are not available in precomputed PCA mode.")
            else:
                result = run_live_rfm_pca(selected_features, n_components, scale_features)
                loadings_fig = build_loadings_figure(result["loadings"], show_loadings)

            explained_variance_ratio = result["explained_variance_ratio"]
            variance_fig = build_variance_figure(explained_variance_ratio)

            pca_df = result["pca_df"]
            if int(n_components) >= 3 and "PC3" in pca_df.columns:
                scatter_fig = build_pca_scatter_3d(pca_df, color_col="PC1")
            else:
                scatter_fig = build_pca_scatter_2d(pca_df, color_col="PC1")

            cumulative_explained = explained_variance_ratio.sum()

            summary_items = [
                html.P(f"PCA Source: {result['mode_label']}"),
                html.P(f"Number of observations used: {result['n_observations']:,}"),
                html.P(f"Number of selected features: {len(result['features_used'])}"),
                html.P(f"Features used: {', '.join(result['features_used'])}"),
            ]

            for i, ratio in enumerate(explained_variance_ratio, start=1):
                summary_items.append(html.P(f"PC{i} Variance Explained: {ratio * 100:.2f}%"))

            summary_items.extend(
                [
                    html.Hr(),
                    html.P(f"Total Variance Explained: {cumulative_explained * 100:.2f}%"),
                    html.P(f"Scaling applied: {'Yes' if scale_features else 'No'}"),
                    html.P(f"Loadings shown: {'Yes' if (show_loadings and source_mode != 'rfm_pca') else 'No'}"),
                    html.P("Insight: PCA reduced customer RFM behavior into a smaller set of orthogonal components."),
                ]
            )

            return (
                "PCA completed successfully.",
                "success",
                html.Div(summary_items),
                variance_fig,
                scatter_fig,
                loadings_fig,
            )

        except Exception as e:
            return (
                f"PCA failed: {str(e)}",
                "danger",
                html.P(str(e)),
                empty_figure("Explained Variance"),
                empty_figure("PCA Graph"),
                empty_figure("PCA Loadings"),
            )