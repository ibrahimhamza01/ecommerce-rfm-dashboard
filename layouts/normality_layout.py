from dash import dcc, html
import dash_bootstrap_components as dbc

layout = dbc.Container(
    [
        # 🔹 TOP: CONTROLS (FULL WIDTH)
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Statistical Parameters"),
                            dbc.CardBody(
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Label("Select Feature"),
                                                dbc.Select(
                                                    id="normality-feature-select",
                                                    options=[],
                                                    value=None,
                                                ),
                                            ],
                                            md=4,
                                        ),

                                        dbc.Col(
                                            [
                                                dbc.Label("Select Tests"),
                                                dbc.Checklist(
                                                    id="normality-test-checklist",
                                                    options=[
                                                        {"label": "Shapiro-Wilk", "value": "shapiro"},
                                                        {"label": "Kolmogorov-Smirnov", "value": "ks"},
                                                        {"label": "D’Agostino K²", "value": "dagostino"},
                                                    ],
                                                    value=["shapiro", "ks", "dagostino"],
                                                    inline=True,
                                                ),
                                            ],
                                            md=5,
                                        ),

                                        dbc.Col(
                                            dbc.Button(
                                                "Run Normality Test",
                                                id="run-normality-btn",
                                                color="success",
                                                className="w-100 mt-4",
                                            ),
                                            md=3,
                                        ),
                                    ]
                                )
                            ),
                        ],
                        className="shadow-sm mb-3",
                    ),
                    width=12,
                )
            ]
        ),

        # 🔹 RESULTS + PLOTS
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Test Statistics"),
                            dbc.CardBody(
                                [
                                    dbc.Alert(
                                        id="normality-alert",
                                        children="No normality test has been run yet.",
                                        color="info",
                                    ),
                                    html.Div(id="normality-summary-text"),
                                ]
                            ),
                        ],
                        className="shadow-sm mb-3",
                    ),
                    md=4,
                ),

                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Feature Distribution"),
                            dbc.CardBody(
                                dcc.Graph(id="normality-distribution-graph")
                            ),
                        ],
                        className="shadow-sm mb-3",
                    ),
                    md=4,
                ),

                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Normal Q-Q Plot"),
                            dbc.CardBody(
                                dcc.Graph(id="normality-qq-graph")
                            ),
                        ],
                        className="shadow-sm mb-3",
                    ),
                    md=4,
                ),
            ]
        ),
    ],
    fluid=True,
)