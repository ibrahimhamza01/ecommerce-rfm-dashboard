from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Statistics Controls"),
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Label("Data Source"),
                                                    dbc.RadioItems(
                                                        id="stats-data-source",
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
                                                    dbc.Label("Numeric Columns"),
                                                    dcc.Dropdown(
                                                        id="stats-numeric-columns",
                                                        options=[],
                                                        value=[],
                                                        multi=True,
                                                        placeholder="Select numeric columns",
                                                    ),
                                                ],
                                                md=4,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Correlation Method"),
                                                    dbc.Select(
                                                        id="stats-corr-method",
                                                        options=[
                                                            {"label": "Pearson", "value": "pearson"},
                                                            {"label": "Spearman", "value": "spearman"},
                                                            {"label": "Kendall", "value": "kendall"},
                                                        ],
                                                        value="pearson",
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Hypothesis Test"),
                                                    dbc.Select(
                                                        id="stats-test-type",
                                                        options=[
                                                            {"label": "One-Sample T-Test", "value": "one_sample_t"},
                                                            {"label": "Two-Sample T-Test by Category", "value": "two_sample_t"},
                                                            {"label": "ANOVA by Category", "value": "anova"},
                                                            {"label": "Chi-Square Test", "value": "chi_square"},
                                                        ],
                                                        value="one_sample_t",
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Run"),
                                                    dbc.Button(
                                                        "Run Statistics",
                                                        id="run-stats-btn",
                                                        color="success",
                                                        className="w-100",
                                                        n_clicks=0,
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                        ]
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Label("Test Numeric Column"),
                                                    dbc.Select(
                                                        id="stats-test-numeric",
                                                        options=[],
                                                        value=None,
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Test Category 1"),
                                                    dbc.Select(
                                                        id="stats-test-category-1",
                                                        options=[],
                                                        value=None,
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Test Category 2"),
                                                    dbc.Select(
                                                        id="stats-test-category-2",
                                                        options=[],
                                                        value=None,
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Hypothesized Mean"),
                                                    dbc.Input(
                                                        id="stats-hypothesized-mean",
                                                        type="number",
                                                        value=0,
                                                        step=0.01,
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                        ],
                                        className="mt-3",
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
                            dbc.CardHeader("Key Statistical Summary"),
                            dbc.CardBody(
                                [
                                    dbc.Alert(
                                        id="stats-alert",
                                        children="Run statistics to generate results.",
                                        color="info",
                                        className="mb-3",
                                    ),
                                    html.Div(id="stats-summary-text"),
                                    html.Hr(),
                                    dbc.Button(
                                        "Download Descriptive Statistics",
                                        id="download-stats-btn",
                                        color="primary",
                                        outline=True,
                                        className="w-100",
                                        n_clicks=0,
                                    ),
                                    dcc.Download(id="download-stats-csv"),
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
                            dbc.CardHeader("Correlation Heatmap"),
                            dbc.CardBody(
                                dcc.Graph(id="stats-correlation-graph")
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
                            dbc.CardHeader("Descriptive Statistics"),
                            dbc.CardBody(
                                dash_table.DataTable(
                                    id="stats-descriptive-table",
                                    columns=[],
                                    data=[],
                                    page_size=10,
                                    sort_action="native",
                                    filter_action="native",
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "textAlign": "left",
                                        "padding": "8px",
                                        "fontSize": "12px",
                                    },
                                    style_header={
                                        "fontWeight": "bold",
                                        "backgroundColor": "#f8f9fa",
                                    },
                                )
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
                            dbc.CardHeader("Hypothesis Test Results"),
                            dbc.CardBody(
                                html.Div(id="stats-test-results")
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