from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
from cluster_sim.simulator import ClusterState
import dash_cytoscape as cyto
from components import (
    move_log,
    plotoptions,
)
from components.components_2d import qubit_panel, postprocess_cyto_data_elements
from typing import List, Dict, Any
import random

import logging

# Initialize the figure of the user's browsing section
def _init_state():
    G = ClusterState(5)
    for i in range(5):
        G.H(i)

    cyto_data = G.to_cytoscape(export_elements=True)

    positions= [{'x': random.randint(0, 300), 'y': random.randint(0, 300)} for i in range(len(G))]

    cyto_data = postprocess_cyto_data_elements(cyto_data, positions)

    return cyto_data
    

figure_2d = cyto.Cytoscape(
    id="figure-app", 
    layout={"name": "random"}, 
    style={"width": "100%", "height": "100%"}, 
    stylesheet=[
        {
            'selector': 'node',
            'style': {
                'label': 'data(vop)'
            }
        }],
    elements= _init_state(),
    selectedNodeData=[]
)

layout_dropdown = dcc.Dropdown(
    ["random", "grid", "circle", "breadthfirst", "cose", "concentric", "preset"],
    "preset",
    id="graph_layout_dropdown",
)

tab_1 = dbc.Col(
    [
        qubit_panel,
        # display_options
    ],
)

tab_3 = dbc.Col(
    [
        move_log,
    ]
)

tab_5 = dbc.Col(
    [
        plotoptions,
        layout_dropdown
    ]
)

tab_ui_2d = html.Div(
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
                dbc.Tab(tab_5, label="Plot Options", tab_id="tab-5"),
            ],
            id="tabs",
            active_tab="tab-1",
        ),
    ]
)



@callback(
    Output("ui", "children", allow_duplicate=True),
    Input("figure-app", "selectedNodeData"),
    prevent_initial_call=True,
)
def displaySelectedNodeData(data_list : List[Dict[str, Any]]):
    if not data_list:
        return "Click on the graph to select nodes, or SHIFT+click to select multiple nodes."
    else:
        logging.debug(f"Selected {[i["value"] for i in data_list]}")
        return f"Selected {[i["value"] for i in data_list]}"

@callback(
    Output("figure-app", "layout"),
    Input("graph_layout_dropdown", "value"),
    prevent_initial_call=True,
)
def update_layout(value):
    return {"name": value}
