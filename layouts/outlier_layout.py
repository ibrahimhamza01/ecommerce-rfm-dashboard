from dash import dcc, html
import dash_bootstrap_components as dbc

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Outlier Controls"),
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Label("Detection Method"),
                                                    dbc.Select(
                                                        id="outlier-method-select",
                                                        options=[
                                                            {"label": "IQR", "value": "iqr"},
                                                            {"label": "Z-Score", "value": "zscore"},
                                                            {"label": "Isolation Forest", "value": "isolation_forest"},
                                                        ],
                                                        value="iqr",
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Select Numeric Features"),
                                                    dbc.Checklist(
                                                        id="outlier-feature-checklist",
                                                        options=[],
                                                        value=[],
                                                        inline=False,
                                                    ),
                                                ],
                                                md=4,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label(
                                                        id="outlier-threshold-label",
                                                        children="Sensitivity Threshold (IQR Multiplier): 1.50",
                                                    ),
                                                    dcc.Slider(
                                                        id="outlier-threshold-slider",
                                                        min=0.5,
                                                        max=5.0,
                                                        step=0.1,
                                                        value=1.5,
                                                        marks={
                                                            0.5: "0.5",
                                                            1.0: "1.0",
                                                            1.5: "1.5",
                                                            2.0: "2.0",
                                                            3.0: "3.0",
                                                            4.0: "4.0",
                                                            5.0: "5.0",
                                                        },
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Apply"),
                                                    dbc.Button(
                                                        "Re-Calculate Outliers",
                                                        id="apply-outlier-btn",
                                                        color="success",
                                                        className="w-100",
                                                        n_clicks=0,
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
                            dbc.CardHeader("Detection Results"),
                            dbc.CardBody(
                                [
                                    dbc.Alert(
                                        id="outlier-alert",
                                        children="No outlier detection has been applied yet.",
                                        color="info",
                                        className="mb-3",
                                    ),
                                    html.Div(id="outlier-summary-text"),
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
                            dbc.CardHeader("Before vs After Boxplots"),
                            dbc.CardBody(
                                dcc.Graph(id="outlier-boxplot-graph")
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
                            dbc.CardHeader("Outliers Removed by Feature"),
                            dbc.CardBody(
                                dcc.Graph(id="outlier-count-graph")
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    md=6,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.Div("Distribution Comparison", className="fw-semibold"),
                                            md=6,
                                        ),
                                        dbc.Col(
                                            dbc.Select(
                                                id="distribution-feature-select",
                                                options=[],
                                                value=None,
                                            ),
                                            md=6,
                                        ),
                                    ],
                                    className="g-2 align-items-center",
                                )
                            ),
                            dbc.CardBody(
                                dcc.Graph(id="outlier-distribution-graph")
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    md=6,
                ),
            ],
            className="mt-3 mb-4",
        ),
    ],
    fluid=True,
)