import dash_bootstrap_components as dbc
from dash import html

from components import (
    move_log,
    reset_graph,
    algorithms,
    hover_data,
    zoom_data,
    load_graph,
    measurementbasis,
    display_options,
    error_channel,
    stabilizer,
    settings,
)
from dash import dcc


tab_1 = dbc.Col(
    [
        measurementbasis,
        reset_graph,
        display_options,
    ],
)

tab_2 = dbc.Col(
    [
        error_channel,
        algorithms,
    ]
)

tab_3 = dbc.Col(
    [
        load_graph,
        move_log,
        hover_data,
        zoom_data,
    ]
)

tab_4 = dbc.Col([stabilizer])

tab_5 = dbc.Col([settings])

tab_ui = html.Div(
    [
        dbc.CardBody(
            [
                html.H2("Cluster Sim"),
                dcc.Loading(
                    dbc.Alert(
                        "Click on the graph to measure nodes.",
                        color="primary",
                        id="ui",
                    ),
                    delay_show=100,
                    delay_hide=100,
                    custom_spinner=dbc.Spinner(color="primary"),
                ),
            ]
        ),
        html.Hr(),
        dbc.Tabs(
            [
                dbc.Tab(tab_1, label="Reset Graph", tab_id="tab-1"),
                dbc.Tab(tab_2, label="Algorithms", tab_id="tab-2"),
                dbc.Tab(tab_3, label="Hover Data", tab_id="tab-3"),
                dbc.Tab(tab_4, label="Stabilizers", tab_id="tab-4"),
                dbc.Tab(tab_5, label="Settings", tab_id="tab-5"),
            ],
            id="tabs",
            active_tab="tab-1",
        ),
    ]
)
