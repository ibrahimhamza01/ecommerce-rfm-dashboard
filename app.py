from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

from layouts.load_data_layout import layout as load_data_layout
from callbacks.load_data_callbacks import register_callbacks as register_load_data_callbacks

from layouts.data_cleaning_layout import layout as data_cleaning_layout
from callbacks.data_cleaning_callbacks import register_callbacks as register_data_cleaning_callbacks

from layouts.outlier_layout import layout as outlier_layout
from callbacks.outlier_callbacks import register_callbacks as register_outlier_callbacks

from layouts.normality_layout import layout as normality_layout
from callbacks.normality_callbacks import register_callbacks as register_normality_callbacks

from layouts.transformation_layout import layout as transformation_layout
from callbacks.transformation_callbacks import register_callbacks as register_transformation_callbacks

from layouts.pca_layout import layout as pca_layout
from callbacks.pca_callbacks import register_callbacks as register_pca_callbacks

from layouts.numerical_layout import layout as numerical_layout
from callbacks.numerical_callbacks import register_callbacks as register_numerical_callbacks

from layouts.categorical_layout import layout as categorical_layout
from callbacks.categorical_callbacks import register_callbacks as register_categorical_callbacks

from layouts.statistics_layout import layout as statistics_layout
from callbacks.statistics_callbacks import register_callbacks as register_statistics_callbacks

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

app.title = "E-Commerce Customer Behavior & RFM Analytics"
server = app.server


app.layout = dbc.Container(
    [
        dcc.Store(id="raw-data-store"),
        dcc.Store(id="current-data-store"),
        dcc.Store(id="metadata-store"),
        dcc.Store(id="history-store", data=[]),

        # CLEAN HEADER
        html.Div(
            [
                html.H2(
                    "E-Commerce Customer Behavior & RFM Analytics Dashboard",
                    className="text-center text-white fw-bold",
                    style={"margin": "0"},
                ),
                html.P(
                    "Customer Segmentation • Statistical Analysis • Interactive Visualization",
                    className="text-center text-light",
                    style={"margin": "0", "fontSize": "14px"},
                ),
            ],
            style={
                "backgroundColor": "#1f2c3d",
                "padding": "12px",
                "borderBottom": "2px solid #0d6efd",
                "marginBottom": "15px",
            },
        ),

        dbc.Tabs(
            id="main-tabs",
            active_tab="tab-load-data",
            children=[
                dbc.Tab(label="Load Data", tab_id="tab-load-data"),
                dbc.Tab(label="Data Cleaning", tab_id="tab-data-cleaning"),
                dbc.Tab(label="Outlier Detection", tab_id="tab-outlier-detection"),
                dbc.Tab(label="Normality Tests", tab_id="tab-normality-tests"),
                dbc.Tab(label="Data Transformation", tab_id="tab-data-transformation"),
                dbc.Tab(label="PCA", tab_id="tab-pca"),
                dbc.Tab(label="Numerical Plots", tab_id="tab-numerical-plots"),
                dbc.Tab(label="Categorical Plots", tab_id="tab-categorical-plots"),
                dbc.Tab(label="Statistics", tab_id="tab-statistics"),
            ],
            className="mb-3",
        ),

        html.Div(id="tab-content"),
    ],
    fluid=True,
    className="p-0",
)


@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab"),
)
def render_tab_content(active_tab):
    if active_tab == "tab-load-data":
        return load_data_layout
    if active_tab == "tab-data-cleaning":
        return data_cleaning_layout
    if active_tab == "tab-outlier-detection":
        return outlier_layout
    if active_tab == "tab-normality-tests":
        return normality_layout
    if active_tab == "tab-data-transformation":
        return transformation_layout
    if active_tab == "tab-pca":
        return pca_layout
    if active_tab == "tab-numerical-plots":
        return numerical_layout
    if active_tab == "tab-categorical-plots":
        return categorical_layout
    if active_tab == "tab-statistics":
        return statistics_layout

    return dbc.Alert("Tab not found.", color="danger")


register_load_data_callbacks(app)
register_data_cleaning_callbacks(app)
register_outlier_callbacks(app)
register_normality_callbacks(app)
register_transformation_callbacks(app)
register_pca_callbacks(app)
register_numerical_callbacks(app)
register_categorical_callbacks(app)
register_statistics_callbacks(app)


if __name__ == "__main__":
    app.run(debug=True)