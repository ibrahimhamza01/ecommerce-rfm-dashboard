from dash import dcc, html
import dash_bootstrap_components as dbc

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Numerical Plot Controls"),
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Label("Data Source"),
                                                    dbc.RadioItems(
                                                        id="numerical-data-source",
                                                        options=[
                                                            {"label": "Current Dataset", "value": "current"},
                                                            {"label": "RFM Dataset", "value": "rfm"},
                                                        ],
                                                        value="current",
                                                        inline=False,
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Plot Type"),
                                                    dbc.Select(
                                                        id="numerical-plot-type",
                                                        options=[
                                                            {"label": "Histogram", "value": "histogram"},
                                                            {"label": "Histogram + KDE", "value": "hist_kde"},
                                                            {"label": "Box Plot", "value": "box"},
                                                            {"label": "Violin Plot", "value": "violin"},
                                                            {"label": "Line Plot", "value": "line"},
                                                            {"label": "Area Plot", "value": "area"},
                                                            {"label": "Scatter Plot", "value": "scatter"},
                                                            {"label": "Regression Plot", "value": "regression"},
                                                            {"label": "Hexbin / Density Heatmap", "value": "hexbin"},
                                                            {"label": "Contour Plot", "value": "contour"},
                                                            {"label": "3D Scatter", "value": "scatter3d"},
                                                            {"label": "Q-Q Plot", "value": "qq"},
                                                            {"label": "Correlation Heatmap", "value": "heatmap"},
                                                        ],
                                                        value="histogram",
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("X Feature"),
                                                    dbc.Select(
                                                        id="numerical-x-feature",
                                                        options=[],
                                                        value=None,
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Y Feature"),
                                                    dbc.Select(
                                                        id="numerical-y-feature",
                                                        options=[],
                                                        value=None,
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Z Feature"),
                                                    dbc.Select(
                                                        id="numerical-z-feature",
                                                        options=[],
                                                        value=None,
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Bins / Resolution"),
                                                    dcc.Slider(
                                                        id="numerical-bins-slider",
                                                        min=10,
                                                        max=100,
                                                        step=5,
                                                        value=40,
                                                        marks={
                                                            10: "10",
                                                            25: "25",
                                                            40: "40",
                                                            60: "60",
                                                            80: "80",
                                                            100: "100",
                                                        },
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    width=12,
                )
            ],
            className="mt-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Plot Summary"),
                            dbc.CardBody(
                                [
                                    dbc.Alert(
                                        id="numerical-alert",
                                        children="Select a plot type and features to explore numerical data.",
                                        color="info",
                                        className="mb-3",
                                    ),
                                    html.Div(id="numerical-summary-text"),
                                ]
                            ),
                        ],
                        className="shadow-sm h-100",
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Numerical Plot"),
                            dbc.CardBody(
                                dcc.Graph(id="numerical-main-graph")
                            ),
                        ],
                        className="shadow-sm h-100",
                    ),
                    md=8,
                ),
            ],
            className="mt-3 mb-4",
        ),
    ],
    fluid=True,
)