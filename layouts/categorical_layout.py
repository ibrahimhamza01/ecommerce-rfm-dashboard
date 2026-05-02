from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_daq as daq

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Categorical Plot Controls"),
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Label("Data Source"),
                                                    dbc.RadioItems(
                                                        id="categorical-data-source",
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
                                                        id="categorical-plot-type",
                                                        options=[
                                                            {"label": "Count Plot", "value": "count"},
                                                            {"label": "Bar Plot", "value": "bar"},
                                                            {"label": "Grouped Bar Plot", "value": "grouped_bar"},
                                                            {"label": "Stacked Bar Plot", "value": "stacked_bar"},
                                                            {"label": "Pie Chart", "value": "pie"},
                                                            {"label": "Heatmap", "value": "heatmap"},
                                                            {"label": "Strip Plot", "value": "strip"},
                                                            {"label": "Swarm-like Plot", "value": "swarm"},
                                                        ],
                                                        value="count",
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Primary Category"),
                                                    dbc.Select(
                                                        id="categorical-x-feature",
                                                        options=[],
                                                        value=None,
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Secondary Category"),
                                                    dbc.Select(
                                                        id="categorical-y-feature",
                                                        options=[],
                                                        value=None,
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Numeric Feature"),
                                                    dbc.Select(
                                                        id="categorical-value-feature",
                                                        options=[],
                                                        value=None,
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Normalize"),
                                                    daq.BooleanSwitch(
                                                        id="categorical-normalize-switch",
                                                        on=False,
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
                                                    dbc.Label("Top Categories"),
                                                    daq.Knob(
                                                        id="categorical-topn-knob",
                                                        min=3,
                                                        max=20,
                                                        value=10,
                                                        label="Top N",
                                                        size=120,
                                                        scale={
                                                            "start": 3,
                                                            "interval": 1,
                                                            "labelInterval": 4,
                                                        },
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Plot Color"),
                                                    daq.ColorPicker(
                                                        id="categorical-color-picker",
                                                        label="Pick Color",
                                                        value={"hex": "#1f77b4"},
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Rows Used"),
                                                    daq.LEDDisplay(
                                                        id="categorical-rows-led",
                                                        value="0",
                                                        size=24,
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Unique Categories"),
                                                    daq.Gauge(
                                                        id="categorical-unique-gauge",
                                                        min=0,
                                                        max=50,
                                                        value=0,
                                                        showCurrentValue=True,
                                                    ),
                                                ],
                                                md=2,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Missing %"),
                                                    daq.GraduatedBar(
                                                        id="categorical-missing-bar",
                                                        min=0,
                                                        max=100,
                                                        value=0,
                                                        showCurrentValue=True,
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
                            dbc.CardHeader("Categorical Summary"),
                            dbc.CardBody(
                                [
                                    dbc.Alert(
                                        id="categorical-alert",
                                        children="Select a categorical plot to begin.",
                                        color="info",
                                        className="mb-3",
                                    ),
                                    html.Div(id="categorical-summary-text"),
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
                            dbc.CardHeader("Categorical Plot"),
                            dbc.CardBody(
                                dcc.Graph(id="categorical-main-graph")
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