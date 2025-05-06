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
    figure,
    error_channel,
)


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


tab_ui = html.Div(
    [
        dbc.CardBody(
            [
                html.H2("Cluster Sim"),
                dbc.Alert(
                    "Click on the graph to measure nodes. Dismiss this message to hide alerts.",
                    color="primary",
                    id="ui",
                    dismissable=True,
                ),
            ]
        ),
        html.Hr(),
        dbc.Tabs(
            [
                dbc.Tab(tab_1, label="Reset Graph", tab_id="tab-1"),
                dbc.Tab(tab_2, label="Algorithms", tab_id="tab-2"),
                dbc.Tab(tab_3, label="Hover Data", tab_id="tab-3"),
            ],
            id="tabs",
            active_tab="tab-1",
        ),
    ]
)
