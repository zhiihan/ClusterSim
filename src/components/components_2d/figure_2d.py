from dash import dcc, html, callback, Input, Output
from textwrap import dedent as d
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

    for i in range(5):
        G.CZ(i, (i+1)%5)

    cyto_data = G.to_cytoscape(export_elements=True)
    positions= [{'x': random.randint(0, 300), 'y': random.randint(0, 300)} for i in range(len(G))]
    cyto_data = postprocess_cyto_data_elements(cyto_data, positions)
    return cyto_data
    

figure_2d = cyto.Cytoscape(
    id="figure-app", 
    layout={"name": "preset"}, 
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

tab_6 = dbc.Col(
    [
        dbc.Card(
    dbc.CardBody(dcc.Markdown(
                                d("""
                    **Local Clifford**

                    Local Clifford gates have the following decomposition:

                        ```
                        local_clifford = {
                            'IA': 'I',
                            'XA': 'X', # HSSH
                            'YA': 'Y', # SSHSSH
                            'ZA': 'Z', # SS
                            'IB': 'HSSHS', 
                            'XB': 'SSS',
                            'YB': 'S',
                            'ZB': 'HSSHSS',
                            'IC': 'HSSHSSH',
                            'XC': 'HSS',
                            'YC': 'H',
                            'ZC': 'SSH',
                            'ID': 'SSSHS',
                            'XD': 'SSHSH',
                            'YD': 'SHS',
                            'ZD': 'HSH',
                            'IE': 'SHSH',
                            'XE': 'SSHS',
                            'YE': 'SSSHSH',
                            'ZE': 'HS',
                            'IF': 'SH',
                            'YF': 'SHSSHSSH',
                            'XF': 'SSSH',
                            'ZF': 'SHSS',
                        }
                        ```
                        """)
        )))
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
                dbc.Tab(tab_6, label="Information", tab_id="tab-6"),
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



@callback(
    Output("figure-app", "stylesheet"),
    Input("figure-app", "elements"),
)
def update_stylesheet(_):
    
    label_style  = [{
        'selector': 'node',
        'style': {
            'label': 'data(vop)'
        }
    }]

    # Operators are applied from right to left. (right applied first)
    local_clifford = {
        'IA': 'I',
        'XA': 'X', # HSSH
        'YA': 'Y', # SSHSSH
        'ZA': 'Z', # SS
        'IB': 'HSSHS', 
        'XB': 'SSS',
        'YB': 'S',
        'ZB': 'HSSHSS',
        'IC': 'HSSHSSH',
        'XC': 'HSS',
        'YC': 'H',
        'ZC': 'SSH',
        'ID': 'SSSHS',
        'XD': 'SSHSH',
        'YD': 'SHS',
        'ZD': 'HSH',
        'IE': 'SHSH',
        'XE': 'SSHS',
        'YE': 'SSSHSH',
        'ZE': 'HS',
        'IF': 'SH',
        'YF': 'SHSSHSSH',
        'XF': 'SSSH',
        'ZF': 'SHSS',
    }

    color_palette = {
        "IA": "#93C5FD",
        "IB": "#60A5FA",
        "IC": "#3B82F6",
        "ID": "#2563EB",
        "IE": "#1D4ED8",
        "IF": "#1F3A8A",

        "XA": "#6EE7B7",
        "XB": "#34D399",
        "XC": "#10B981",
        "XD": "#059669",
        "XE": "#166534",
        "XF": "#14532D",

        "YA": "#FCD34D",
        "YB": "#FBBF24",
        "YC": "#F59E0B",
        "YD": "#EA580C",
        "YE": "#C2410C",
        "YF": "#9A3412",

        "ZA": "#DDD6FE",
        "ZB": "#C084FC",
        "ZC": "#A855F7",
        "ZD": "#9333EA",
        "ZE": "#7E22CE",
        "ZF": "#581C87",
    }

    color_styles = []
    for label, color in color_palette.items():
        color_styles.append({
            "selector": f'[vop *= "{label}"]',
            "style": {
                "background-color": color, 
                "border-width": 2,
                "border-color": "#000000",
            },
        })


    selected_style = [
        {
            "selector": "node:selected",
            "style": {
                "border-width": 3,
                "border-color": "blue",
            },
        }]

    return label_style + color_styles + selected_style