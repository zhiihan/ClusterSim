from dash import dcc, html, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
from cluster_sim.simulator import ClusterState
from cluster_sim.app import BrowserState, update_plot_cytoscape
import dash_cytoscape as cyto
from components import (
    move_log,
    plotoptions,
)
from components.components_2d import qubit_panel
from typing import List, Dict, Any

import logging

# Initialize the figure of the user's browsing section
def _init_state():
    G = ClusterState(5)
    cyto_data = G.to_cytoscape()

    return cyto_data["elements"]
    

figure_2d = cyto.Cytoscape(
    id="figure-app", 
    layout={"name": "random"}, 
    style={"width": "100%", "height": "100%"}, 
    elements=_init_state()
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
    Output("figure-app", "elements"),
    Input("draw-plot", "data"),
    State("browser-data", "data"),
    State("graph-data", "data"),
)
def draw_plot(draw_plot, browser_data, graphData):
    """
    Called when ever the plot needs to be drawn.
    """
    if browser_data is None:
        return no_update

    browser_state = BrowserState.from_json(browser_data)
    G = ClusterState.from_json(graphData)

    fig = update_plot_cytoscape(browser_state, G)
    # Make sure the view/angle stays the same when updating the figure
    return fig


@callback(
    Output("ui", "children", allow_duplicate=True),
    Input("figure-app", "selectedNodeData"),
    prevent_initial_call=True,
)
def displaySelectedNodeData(data_list : List[Dict[str, Any]]):
    if not data_list:
        return "Click on the graph to select nodes, or SHIFT+click to select multiple nodes."
    else:
        logging.debug(data_list)
        return f"Selected {[i["value"] for i in data_list]}"