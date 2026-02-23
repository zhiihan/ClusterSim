import dash_bootstrap_components as dbc
from dash import html

from components import (
    move_log,
    reset_graph,
    hover_data,
    zoom_data,
    load_graph,
    measurementbasis,
    plotoptions,
    stabilizer,
)
from dash import dcc


tab_1 = dbc.Col(
    [
        measurementbasis,
        # display_options
    ],
)

# tab_2 = dbc.Col(
#     [
#         algorithms,
#     ]
# )

tab_3 = dbc.Col(
    [
        reset_graph,
        load_graph,
        move_log,
    ]
)

tab_4 = dbc.Col([stabilizer])

tab_5 = dbc.Col(
    [
        plotoptions,
    ]
)

tab_6 = dbc.Col([
    hover_data,
    zoom_data,
])

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
                dbc.Tab(tab_1, label="Measurements", tab_id="tab-1"),
                # dbc.Tab(tab_2, label="Algorithms", tab_id="tab-2"),
                dbc.Tab(tab_3, label="Reset and Load", tab_id="tab-3"),
                dbc.Tab(tab_4, label="Stabilizers", tab_id="tab-4"),
                dbc.Tab(tab_5, label="Plot Options", tab_id="tab-5"),
                dbc.Tab(tab_6, label="Debug", tab_id="tab-6"),
            ],
            id="tabs",
            active_tab="tab-1",
        ),
    ]
)
