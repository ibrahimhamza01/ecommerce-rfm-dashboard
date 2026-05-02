from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Cleaning Parameters"),
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Label("Missing Value Column"),
                                                    dbc.Select(
                                                        id="clean-missing-column",
                                                        options=[],
                                                        value=None,
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Missing Value Treatment"),
                                                    dbc.Select(
                                                        id="clean-missing-select",
                                                        options=[
                                                            {"label": "Keep Missing Values", "value": "keep"},
                                                            {"label": "Drop Rows with Missing Values", "value": "drop_rows"},
                                                            {"label": "Forward Fill", "value": "ffill"},
                                                            {"label": "Backward Fill", "value": "bfill"},
                                                            {"label": "Fill with Mean (numeric only)", "value": "mean"},
                                                            {"label": "Fill with Median (numeric only)", "value": "median"},
                                                            {"label": "Fill with Mode", "value": "mode"},
                                                            {"label": "Fill with Constant", "value": "constant"},
                                                            {"label": "Add Missing Flag Column", "value": "flag_missing"},
                                                        ],
                                                        value="keep",
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Constant Fill Value"),
                                                    dbc.Input(
                                                        id="clean-constant-value",
                                                        type="text",
                                                        placeholder="Used only for constant fill",
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Add Rule"),
                                                    dbc.Button(
                                                        "Add Rule",
                                                        id="add-missing-rule-btn",
                                                        color="primary",
                                                        className="w-100",
                                                        n_clicks=0,
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Clear Rules"),
                                                    dbc.Button(
                                                        "Clear Rules",
                                                        id="clear-missing-rules-btn",
                                                        color="danger",
                                                        outline=True,
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
                                                    dbc.Label("Duplicate Handling"),
                                                    dbc.Select(
                                                        id="clean-duplicate-select",
                                                        options=[
                                                            {"label": "Keep Duplicates", "value": "keep"},
                                                            {"label": "Remove Exact Duplicates", "value": "remove_exact"},
                                                        ],
                                                        value="keep",
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Administrative Filter"),
                                                    dbc.Select(
                                                        id="clean-admin-select",
                                                        options=[
                                                            {"label": "Keep Administrative Entries", "value": "keep"},
                                                            {"label": "Remove Manual / Bad Debt / Adjust Entries", "value": "remove_admin"},
                                                        ],
                                                        value="keep",
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Feature Engineering"),
                                                    dbc.Checklist(
                                                        id="feature-engineering-checklist",
                                                        options=[
                                                            {"label": "Create LineTotal", "value": "linetotal"},
                                                            {"label": "Create TransactionStatus", "value": "transaction_status"},
                                                            {"label": "Create PurchaseQuarter", "value": "purchase_quarter"},
                                                            {"label": "Create PriceCategory", "value": "price_category"},
                                                        ],
                                                        value=[],
                                                        inline=False,
                                                    ),
                                                ],
                                                md=4,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Apply"),
                                                    dbc.Button(
                                                        "Apply Cleaning",
                                                        id="apply-cleaning-btn",
                                                        color="success",
                                                        className="w-100",
                                                        n_clicks=0,
                                                    ),
                                                ],
                                                md=2,
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

        dcc.Store(id="missing-rules-store", data=[]),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Cleaning Summary"),
                            dbc.CardBody(
                                [
                                    dbc.Alert(
                                        id="cleaning-alert",
                                        children="No cleaning has been applied yet.",
                                        color="info",
                                        className="mb-3",
                                    ),
                                    html.Div(id="cleaning-summary-text"),
                                ]
                            ),
                        ],
                        className="shadow-sm h-100",
                    ),
                    md=7,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Queued Missing Value Rules"),
                            dbc.CardBody(
                                dash_table.DataTable(
                                    id="missing-rules-table",
                                    columns=[
                                        {"name": "Column", "id": "column"},
                                        {"name": "Method", "id": "method"},
                                        {"name": "Constant Value", "id": "constant_value"},
                                    ],
                                    data=[],
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
                        className="shadow-sm h-100",
                    ),
                    md=5,
                ),
            ],
            className="mt-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Before vs After Cleaning"),
                            dbc.CardBody(dcc.Graph(id="cleaning-comparison-chart")),
                        ],
                        className="shadow-sm",
                    ),
                    md=6,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Transaction Status"),
                            dbc.CardBody(dcc.Graph(id="transaction-status-chart")),
                        ],
                        className="shadow-sm",
                    ),
                    md=6,
                ),
            ],
            className="mt-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Purchase Quarter Distribution"),
                            dbc.CardBody(dcc.Graph(id="purchase-quarter-chart")),
                        ],
                        className="shadow-sm",
                    ),
                    md=6,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Price Category Distribution"),
                            dbc.CardBody(dcc.Graph(id="price-category-chart")),
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