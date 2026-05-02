from dash import dcc, html
import dash_bootstrap_components as dbc

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Scaling Parameters"),
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Label("Base Feature"),
                                                    dbc.Select(
                                                        id="transform-feature-select",
                                                        options=[],
                                                        value=None,
                                                        className="mb-3",
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Transformation Type"),
                                                    dbc.Select(
                                                        id="transform-method-select",
                                                        options=[
                                                            {"label": "Log", "value": "log"},
                                                            {"label": "Box-Cox", "value": "boxcox"},
                                                            {"label": "Standardization", "value": "standardize"},
                                                            {"label": "MinMax Scaling", "value": "minmax"},
                                                        ],
                                                        value="log",
                                                        className="mb-3",
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Transformation Source"),
                                                    dbc.RadioItems(
                                                        id="transform-source-mode",
                                                        options=[
                                                            {"label": "Apply on Original Feature", "value": "original"},
                                                            {"label": "Apply on Latest Transformed Feature", "value": "latest"},
                                                        ],
                                                        value="original",
                                                        inline=False,
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Apply"),
                                                    dbc.Button(
                                                        "Apply Transformation",
                                                        id="apply-transformation-btn",
                                                        color="success",
                                                        className="w-100",
                                                        n_clicks=0,
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                        ]
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Checklist(
                                                    id="transform-options-checklist",
                                                    options=[
                                                        {"label": "Center Data (Subtract Mean)", "value": "center"},
                                                        {"label": "Scale Data (Standard Deviation)", "value": "scale"},
                                                        {"label": "Handle Zero / Negative Values (Add Constant)", "value": "handle_zero"},
                                                    ],
                                                    value=["handle_zero"],
                                                    inline=False,
                                                ),
                                                md=12,
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
                            dbc.CardHeader("Post-Transformation Stats"),
                            dbc.CardBody(
                                [
                                    dbc.Alert(
                                        id="transformation-alert",
                                        children="No transformation has been applied yet.",
                                        color="info",
                                        className="mb-3",
                                    ),
                                    html.Div(id="transformation-summary-text"),
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
                            dbc.CardHeader("Transformed Feature (Before vs After)"),
                            dbc.CardBody(
                                dcc.Graph(id="transformation-distribution-graph")
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
                            dbc.CardHeader("Normal Q-Q Plot"),
                            dbc.CardBody(
                                dcc.Graph(id="transformation-qq-graph")
                            ),
                        ],
                        className="shadow-sm h-100",
                    ),
                    md=6,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Box Plot (Before vs After)"),
                            dbc.CardBody(
                                dcc.Graph(id="transformation-boxplot-graph")
                            ),
                        ],
                        className="shadow-sm h-100",
                    ),
                    md=6,
                ),
            ],
            className="mt-3 mb-4",
        ),
    ],
    fluid=True,
)