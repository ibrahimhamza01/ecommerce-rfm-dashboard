from dash import dcc, dash_table, html
import dash_bootstrap_components as dbc
import dash_daq as daq

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Load Dataset Controls"),
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Label("Select Dataset"),
                                                    dbc.Select(
                                                        id="dataset-selector",
                                                        options=[
                                                            {
                                                                "label": "Online Retail II (Raw)",
                                                                "value": "data/online_retail_II.csv",
                                                            }
                                                        ],
                                                        value="data/online_retail_II.csv",
                                                    ),
                                                ],
                                                md=4,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Number of Rows"),
                                                    daq.LEDDisplay(
                                                        id="rows-display",
                                                        value="0",
                                                        size=32,
                                                        backgroundColor="#f8f9fa",
                                                        color="#2c7be5",
                                                    ),
                                                ],
                                                md=4,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Number of Columns"),
                                                    daq.LEDDisplay(
                                                        id="cols-display",
                                                        value="0",
                                                        size=32,
                                                        backgroundColor="#f8f9fa",
                                                        color="#2c7be5",
                                                    ),
                                                ],
                                                md=4,
                                            ),
                                        ]
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Button(
                                                    "Load Data",
                                                    id="load-data-btn",
                                                    color="primary",
                                                    className="w-100",
                                                    n_clicks=0,
                                                ),
                                                md=4,
                                            ),
                                            dbc.Col(
                                                dbc.Button(
                                                    "Reset Current Data",
                                                    id="reset-data-btn",
                                                    color="secondary",
                                                    className="w-100",
                                                    n_clicks=0,
                                                ),
                                                md=4,
                                            ),
                                            dbc.Col(
                                                dbc.Badge(
                                                    "Awaiting load...",
                                                    id="load-status-badge",
                                                    color="warning",
                                                    className="p-2",
                                                ),
                                                md=4,
                                                className="d-flex align-items-center mt-2 mt-md-0",
                                            ),
                                        ],
                                        className="mt-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Alert(
                                                    id="load-data-alert",
                                                    children="No dataset loaded yet.",
                                                    color="info",
                                                    className="mb-0",
                                                ),
                                                width=12,
                                            )
                                        ],
                                        className="mt-3",
                                    ),
                                ]
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    md=12,
                )
            ],
            className="mt-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Preview Filters"),
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Label("Countries"),
                                                    dcc.Dropdown(
                                                        id="preview-country-filter",
                                                        multi=True,
                                                        placeholder="Select countries",
                                                    ),
                                                ],
                                                md=6,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Quantity Type"),
                                                    dbc.RadioItems(
                                                        id="preview-quantity-type",
                                                        options=[
                                                            {"label": "Orders", "value": "orders"},
                                                            {"label": "Returns", "value": "returns"},
                                                            {"label": "Both", "value": "both"},
                                                        ],
                                                        value="both",
                                                        inline=False,
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Invoice Date"),
                                                    dcc.DatePickerRange(
                                                        id="preview-date-filter",
                                                        display_format="YYYY-MM-DD",
                                                        minimum_nights=0,
                                                        clearable=False,
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    md=12,
                )
            ],
            className="mt-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                html.Div(
                                    "Filtered Dataset Preview (First 10 Rows)",
                                    className="fw-semibold",
                                )
                            ),
                            dbc.CardBody(
                                dash_table.DataTable(
                                    id="data-preview-table",
                                    page_size=10,
                                    sort_action="native",
                                    filter_action="native",
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "textAlign": "left",
                                        "padding": "8px",
                                        "fontSize": "12px",
                                        "minWidth": "120px",
                                        "maxWidth": "240px",
                                        "whiteSpace": "normal",
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
                    md=12,
                )
            ],
            className="mt-3 mb-4",
        ),
    ],
    fluid=True,
)