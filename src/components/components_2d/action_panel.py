from cycler import L
from cluster_sim import ClusterState
from textwrap import dedent as d
from dash import dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
from typing import List, Dict, Any, Tuple
import pprint

qubit_panel = dbc.Card(
    dbc.CardBody(
        [
            dcc.Markdown(
                d(
                """
                **Select Measurement Basis**

                Click points in the graph, then press the button to measure.
                """
                )
            ),
            dbc.ButtonGroup(
                [        
                    dbc.Button("MZ", outline=True, color="primary", id='MZ'),
                    dbc.Button("MY", outline=True, color="primary", id='MY'),
                    dbc.Button("MX", outline=True, color="primary", id='MX'),
                    dbc.Button("LC", outline=True, color="primary", id='LC')
                    ],
                    id='qubit-action-panel'
            )
        ]
    )
)

@callback(
    Output("ui", "children", allow_duplicate=True),
    Output('figure-app', component_property="elements", allow_duplicate=True),
    Input("MZ", "n_clicks"),
    State("figure-app", "selectedNodeData"),
    State('figure-app', component_property="elements"),
    prevent_initial_call=True,
)
def MZ(n_clicks, selected_node_data, cyto_data):
    method_args = {'force': 0, 'basis' : 'Z'}
    return apply_operation_wrapper('measure', selected_node_data, cyto_data, **method_args)

@callback(
    Output("ui", "children", allow_duplicate=True),
    Output('figure-app', component_property="elements", allow_duplicate=True),
    Input("MY", "n_clicks"),
    State("figure-app", "selectedNodeData"),
    State('figure-app', component_property="elements"),
    prevent_initial_call=True,
)
def MY(n_clicks, selected_node_data, cyto_data):
    method_args = {'force': 0, 'basis' : 'Y'}
    return apply_operation_wrapper('measure', selected_node_data, cyto_data, **method_args)

@callback(
    Output("ui", "children", allow_duplicate=True),
    Output('figure-app', component_property="elements", allow_duplicate=True),
    Input("MX", "n_clicks"),
    State("figure-app", "selectedNodeData"),
    State('figure-app', component_property="elements"),
    prevent_initial_call=True,
)
def MX(n_clicks, selected_node_data, cyto_data):
    method_args = {'force': 0, 'basis' : 'X'}
    return apply_operation_wrapper('measure', selected_node_data, cyto_data, **method_args)

@callback(
    Output("ui", "children", allow_duplicate=True),
    Output('figure-app', component_property="elements", allow_duplicate=True),
    Input("LC", "n_clicks"),
    State("figure-app", "selectedNodeData"),
    State('figure-app', component_property="elements"),
    prevent_initial_call=True,
)
def LC(n_clicks, selected_node_data, cyto_data):
    method_args = {}
    return apply_operation_wrapper('local_complementation', selected_node_data, cyto_data, **method_args)

def apply_operation_wrapper(method_name, selected_node_data, cyto_data, **method_args):
    """Take the buttons for all the call backs and apply the corresponding method in the simulator.

    Args:
        operation_name (_type_): ClusterState.method_name
        selected_node_data (_type_): selected_node_data
        cyto_data (_type_): cyto_data directly from figure-app.elements

    Returns:
        _type_: processed cyto_data for use in figure-app.elements
    """
    selected_nodes = [i["value"] for i in selected_node_data]
    
    if not selected_nodes:
        return no_update, no_update

    cyto_data, positions = preprocess_cyto_data_elements(cyto_data)
    g = ClusterState.from_cytoscape(cyto_data)

    outcomes = []

    for i in selected_nodes:
        outcomes.append(getattr(g, method_name)(i, **method_args))

    cyto_data_new = g.to_cytoscape(export_elements = True)
    cyto_data_new = postprocess_cyto_data_elements(cyto_data_new, positions)

    return f'Measured selected nodes {selected_nodes} with outcomes {outcomes}', cyto_data_new

def preprocess_cyto_data_elements(cyto_data : List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    nodes = []
    edges = []
    positions = []
    for item in cyto_data:
        if item.get('data'):
            if item['data'].get('vop'):
                nodes.append(item) # This is a node
                positions.append(item['position'])
            elif item['data'].get('source'): 
                edges.append(item) # This is an edge
        else:
            raise NotImplementedError('Unknown or no data')
    return {'data':[], 'multigraph': False, 'elements': {'nodes': nodes, 'edges': edges}}, positions

def postprocess_cyto_data_elements(cyto_data: Dict[str, Any], positions: List[Dict[str, Any]]):
    for item, pos in zip(cyto_data, positions):
        if item.get('data'):
            if item['data'].get('vop'):
                item['position'] = pos
        else:
            raise NotImplementedError('Unknown or no data')
    return cyto_data
