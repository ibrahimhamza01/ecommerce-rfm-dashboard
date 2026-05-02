from dash import dcc, html
import dash_bootstrap_components as dbc

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("PCA Configuration"),
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Label("PCA Source"),
                                                    dbc.RadioItems(
                                                        id="pca-source-mode",
                                                        options=[
                                                            {"label": "Run PCA on RFM Table", "value": "rfm_table"},
                                                            {"label": "Use Precomputed RFM PCA", "value": "rfm_pca"},
                                                        ],
                                                        value="rfm_table",
                                                        inline=False,
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Components"),
                                                    dbc.Select(
                                                        id="pca-components-select",
                                                        options=[
                                                            {"label": "2D", "value": 2},
                                                            {"label": "3D", "value": 3},
                                                        ],
                                                        value=3,
                                                        className="mb-3",
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("RFM Features"),
                                                    dcc.Dropdown(
                                                        id="pca-feature-select",
                                                        options=[
                                                            {"label": "Recency", "value": "Recency"},
                                                            {"label": "Frequency", "value": "Frequency"},
                                                            {"label": "MonetaryValue", "value": "MonetaryValue"},
                                                        ],
                                                        value=["Recency", "Frequency", "MonetaryValue"],
                                                        multi=True,
                                                        className="mb-3",
                                                    ),
                                                ],
                                                md=4,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Options"),
                                                    dbc.Checklist(
                                                        id="pca-options-checklist",
                                                        options=[
                                                            {"label": "Scale Features", "value": "scale"},
                                                            {"label": "Show Loadings", "value": "show_loadings"},
                                                        ],
                                                        value=["scale", "show_loadings"],
                                                        inline=False,
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Run"),
                                                    dbc.Button(
                                                        "Run PCA",
                                                        id="run-pca-btn",
                                                        color="success",
                                                        className="w-100",
                                                        n_clicks=0,
                                                    ),
                                                ],
                                                md=1,
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
                            dbc.CardHeader("Variance Explained"),
                            dbc.CardBody(
                                [
                                    dbc.Alert(
                                        id="pca-alert",
                                        children="No PCA has been run yet.",
                                        color="info",
                                        className="mb-3",
                                    ),
                                    html.Div(id="pca-summary-text"),
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
                            dbc.CardHeader("Explained Variance Plot"),
                            dbc.CardBody(
                                dcc.Graph(id="pca-variance-graph")
                            ),
                        ],
                        className="shadow-sm h-100",
                    ),
                    md=8,
                ),
            ],
            className="mt-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("PCA Graph"),
                            dbc.CardBody(
                                dcc.Graph(id="pca-scatter-graph")
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    width=12,
                ),
            ],
            className="mt-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("PCA Loadings"),
                            dbc.CardBody(
                                dcc.Graph(id="pca-loadings-graph")
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    width=12,
                ),
            ],
            className="mt-3 mb-4",
        ),
    ],
    fluid=True,
)