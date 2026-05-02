import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
from scipy.stats import shapiro, kstest, normaltest, probplot


def load_dataset(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    return df


def empty_figure(title: str):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", title=title)
    return fig


def get_numeric_feature_options(file_path: str):
    df = load_dataset(file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    options = [{"label": col, "value": col} for col in numeric_cols]
    default_value = None

    preferred = [col for col in ["Quantity", "Price", "LineTotal"] if col in numeric_cols]
    if preferred:
        default_value = preferred[0]
    elif numeric_cols:
        default_value = numeric_cols[0]

    return options, default_value


def build_distribution_figure(df: pd.DataFrame, feature: str):
    if feature not in df.columns:
        return empty_figure("Feature Distribution")

    values = pd.to_numeric(df[feature], errors="coerce").dropna()

    if values.empty:
        return empty_figure(f"No valid numeric values for {feature}")

    fig = px.histogram(
        x=values,
        nbins=50,
        template="plotly_white",
        title=f"Feature Distribution ({feature})",
        labels={"x": feature, "y": "Count"},
    )

    fig.update_layout(showlegend=False)
    return fig


def build_qq_figure(df: pd.DataFrame, feature: str):
    if feature not in df.columns:
        return empty_figure("Normal Q-Q Plot")

    values = pd.to_numeric(df[feature], errors="coerce").dropna()

    if len(values) < 3:
        return empty_figure(f"Not enough values for Q-Q plot: {feature}")

    osm, osr = probplot(values, dist="norm", fit=False)
    slope, intercept = np.polyfit(osm, osr, 1)

    line_x = np.array(osm)
    line_y = slope * line_x + intercept

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            name="Observed Quantiles",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name="Reference Line",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=f"Normal Q-Q Plot ({feature})",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
    )
    return fig


def run_selected_tests(values: pd.Series, selected_tests: list):
    results = []

    if "shapiro" in selected_tests:
        if len(values) >= 3:
            sample = values
            if len(sample) > 5000:
                sample = sample.sample(5000, random_state=42)
            stat, p_value = shapiro(sample)
            results.append(("Shapiro-Wilk", stat, p_value))
        else:
            results.append(("Shapiro-Wilk", None, None))

    if "ks" in selected_tests:
        if len(values) >= 3:
            mean_val = values.mean()
            std_val = values.std(ddof=1)

            if std_val == 0 or np.isnan(std_val):
                results.append(("Kolmogorov-Smirnov", None, None))
            else:
                standardized = (values - mean_val) / std_val
                stat, p_value = kstest(standardized, "norm")
                results.append(("Kolmogorov-Smirnov", stat, p_value))
        else:
            results.append(("Kolmogorov-Smirnov", None, None))

    if "dagostino" in selected_tests:
        if len(values) >= 8:
            stat, p_value = normaltest(values)
            results.append(("D’Agostino K²", stat, p_value))
        else:
            results.append(("D’Agostino K²", None, None))

    return results


def build_summary(results):
    valid_pvalues = [p for _, _, p in results if p is not None]

    if not valid_pvalues:
        verdict = "Not enough valid results to determine normality."
        verdict_color = "warning"
    elif all(p > 0.05 for p in valid_pvalues):
        verdict = "Fail to reject the null hypothesis. The data appears approximately Gaussian (Normal)."
        verdict_color = "success"
    else:
        verdict = "Reject the null hypothesis. The data is non-Gaussian (Not Normal)."
        verdict_color = "danger"

    result_lines = []
    for test_name, stat, p_value in results:
        if stat is None or p_value is None:
            result_lines.append(html.Li(f"{test_name}: Not enough valid data / test not applicable"))
        else:
            result_lines.append(
                html.Li(f"{test_name} — Statistic: {stat:.2f}, p-value: {p_value:.2f}")
            )

    summary_children = html.Div(
        [
            html.Ul(result_lines),
            html.Hr(),
            html.P(
                [
                    html.Strong("Verdict: "),
                    verdict
                ]
            ),
        ]
    )

    return summary_children, verdict_color


def register_callbacks(app):
    @app.callback(
        [
            Output("normality-feature-select", "options"),
            Output("normality-feature-select", "value"),
        ],
        Input("current-data-store", "data"),
        prevent_initial_call=False,
    )
    def populate_normality_features(current_store):
        if current_store is None or "file_path" not in current_store:
            return [], None

        try:
            return get_numeric_feature_options(current_store["file_path"])
        except Exception:
            return [], None

    @app.callback(
        [
            Output("normality-alert", "children"),
            Output("normality-alert", "color"),
            Output("normality-summary-text", "children"),
            Output("normality-distribution-graph", "figure"),
            Output("normality-qq-graph", "figure"),
        ],
        Input("run-normality-btn", "n_clicks"),
        [
            State("current-data-store", "data"),
            State("normality-feature-select", "value"),
            State("normality-test-checklist", "value"),
        ],
        prevent_initial_call=True,
    )
    def run_normality_analysis(n_clicks, current_store, selected_feature, selected_tests):
        if not n_clicks:
            raise PreventUpdate

        if current_store is None or "file_path" not in current_store:
            raise PreventUpdate

        if not selected_feature:
            return (
                "Please select a feature.",
                "warning",
                html.P("No feature selected."),
                empty_figure("Feature Distribution"),
                empty_figure("Normal Q-Q Plot"),
            )

        if not selected_tests:
            return (
                "Please select at least one test.",
                "warning",
                html.P("No statistical test selected."),
                empty_figure("Feature Distribution"),
                empty_figure("Normal Q-Q Plot"),
            )

        df = load_dataset(current_store["file_path"])

        if selected_feature not in df.columns:
            return (
                "Selected feature was not found in the active dataset.",
                "danger",
                html.P("Invalid feature."),
                empty_figure("Feature Distribution"),
                empty_figure("Normal Q-Q Plot"),
            )

        values = pd.to_numeric(df[selected_feature], errors="coerce").dropna()

        if len(values) < 3:
            return (
                "Not enough valid numeric values to run normality tests.",
                "warning",
                html.P("Insufficient numeric data."),
                empty_figure("Feature Distribution"),
                empty_figure("Normal Q-Q Plot"),
            )

        results = run_selected_tests(values, selected_tests)
        summary_children, verdict_color = build_summary(results)

        distribution_fig = build_distribution_figure(df, selected_feature)
        qq_fig = build_qq_figure(df, selected_feature)

        return (
            "Normality tests completed successfully.",
            verdict_color,
            summary_children,
            distribution_fig,
            qq_fig,
        )