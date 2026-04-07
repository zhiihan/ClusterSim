import itertools
from cluster_sim.simulator import ClusterState
from textwrap import dedent as d
from dash import dcc, callback, Input, Output, State, no_update, callback_context, html
import dash_bootstrap_components as dbc
from typing import Any
import random

qubit_panel = dbc.Card(
    dbc.CardBody(
        [
            dcc.Markdown(
                d(
                    """
                **Select Measurement Basis**

                Click points in the graph, then press the button to measure. 
                
                Forcing measurement only applies if the state is 50/50.
                """
                )
            ),
            dbc.ButtonGroup(
                [
                    dbc.Button("MZ", outline=True, color="primary", id="MZ"),
                    dbc.Button("MY", outline=True, color="primary", id="MY"),
                    dbc.Button("MX", outline=True, color="primary", id="MX"),
                    dbc.Button("Fusion", outline=True, color="primary", id="fusion-gate"),
                ],
            ),
            dbc.Select(
                id="force-measurement",
                options=[
                    {"label": "Force outcome 0 (default)", "value": 0},
                    {"label": "Force outcome 1", "value": 1},
                    {"label": "Random measurement", "value": -1},
                ],
                placeholder="Force outcome 0 (default)",
                value=0,
                style={
                        "minwidth": "300px",
                        "width": "300px"
                    }),
            dbc.Select(
                id="fusion-mode",
                options=[
                    {"label": "XXZZ (default)", "value": 'II'},
                    {"label": "XZZX", "value": 'HI'},
                ],
                placeholder="XXZZ (default)",
                value='II',
                style={
                        "minwidth": "200px",
                        "width": "200px"
                    }),
            html.Br(),
            dcc.Markdown(
                d(
                    """
                **Apply Clifford Gates**

                Click points in the graph, then press the button to apply Clifford gates.
                """
                )
            ),
            dbc.ButtonGroup(
                [
                    dbc.Button("X", outline=True, color="primary", id="X"),
                    dbc.Button("Y", outline=True, color="primary", id="Y"),
                    dbc.Button("Z", outline=True, color="primary", id="Z"),
                    dbc.Button("H", outline=True, color="primary", id="H"),
                    dbc.Button("S", outline=True, color="primary", id="S"),
                    dbc.Button("CX", outline=True, color="primary", id="CX"),
                    dbc.Button("CZ", outline=True, color="primary", id="CZ"),
                ]
            ),
            html.Br(),
            dcc.Markdown(
                d(
                    """
                **Graph operations**

                Click points in the graph, then press the button to apply graph operations.
                """
                )
            ),
            dbc.ButtonGroup(
                [
                    dbc.Button(
                        "Add Node", outline=True, color="primary", id="add-node"
                    ),
                    dbc.Button(
                        "Remove Node", outline=True, color="primary", id="remove-node"
                    ),
                    dbc.Button(
                        "Add Edge", outline=True, color="primary", id="add-edge"
                    ),
                    dbc.Button(
                        "Remove Edge", outline=True, color="primary", id="remove-edge"
                    ),
                    dbc.Button(
                        "Toggle Edge", outline=True, color="primary", id="toggle-edge"
                    ),
                    dbc.Button(
                        "Local Complementation", outline=True, color="primary", id="LC"
                    ),
                ]
            ),
        ]
    )
)

button_operations = {
    # measure buttons with basis
    "MX": ("measure", {"force": 0, "basis": "X"}),
    "MY": ("measure", {"force": 0, "basis": "Y"}),
    "MZ": ("measure", {"force": 0, "basis": "Z"}),
    "fusion-gate": ("fusion_gate", {"force": 0, "mode": "success"}),
    # simple operations with no extra args
    "LC": ("local_complementation", {}),
    "X": ("X", {}),
    "Y": ("Y", {}),
    "Z": ("Z", {}),
    "H": ("H", {}),
    "S": ("S", {}),
    "CX": ("CX", {}),
    "CZ": ("CZ", {}),
    "add-edge": ("add_edge", {}),
    "remove-edge": ("remove_edge", {}),
    "toggle-edge": ("toggle_edge", {}),
    "add-node": ("add_node", {"vop": "YC"}),
    "remove-node": ("remove_node", {}),
}


@callback(
    Output("ui", "children"),
    Output("figure-app", "elements", allow_duplicate=True),
    Output("simulator-representation", "children"),
    Output("move-log", "children"),
    *[Input(btn, "n_clicks") for btn in button_operations.keys()],
    State("fusion-mode", "value"),
    State("force-measurement", "value"),
    State("move-log", "children"),
    State("figure-app", "selectedNodeData"),
    State("figure-app", "elements"),
    prevent_initial_call=True,
)
def handle_buttons(*args):
    log = args[-3]

    # The last two args are selectedNodeData and elements
    selected_node_data = args[-2]
    cyto_data = args[-1]

    # Determine which button triggered the callback
    triggered_id = callback_context.triggered_id
    if not triggered_id or triggered_id not in button_operations:
        raise NotImplementedError

    method_name, method_args = button_operations[triggered_id]
    if method_name == "fusion_gate":
        method_args["gate_control"] = args[-5][0]
        method_args["gate_target"] = args[-5][1]
        method_args["force"] = args[-4]

    elif method_name == "measure":
        method_args["force"] = args[-4]

    selected_nodes = [i["value"] for i in selected_node_data]

    if not selected_nodes and method_name != "add_node":
        return no_update, no_update, no_update, no_update

    return apply_operation_wrapper(
        method_name, selected_nodes, cyto_data, log, method_args, **kwargs
    )


def apply_operation_wrapper(
    method_name: str, selected_nodes: list[int], cyto_data, log, method_args, **kwargs
):
    """Take the buttons for all the call backs and apply the corresponding method in the simulator.

    Args:
        operation_name (str): ClusterState.method_name
        selected_node_data (_type_): selected_node_data
        cyto_data (_type_): cyto_data directly from figure-app.elements

    Returns:
        _type_: processed cyto_data for use in figure-app.elements
    """

    cyto_data, positions = preprocess_cyto_data_elements(cyto_data)
    g = ClusterState.from_cytoscape(cyto_data)

    if method_name in ["add_edge", "remove_edge", "toggle_edge"]:
        for pair in itertools.combinations(selected_nodes, 2):
            getattr(g, method_name)(*pair, **method_args)
        ui = f"Applied {method_name} to {selected_nodes}"
    elif method_name in ["CX", "CZ"]:
        for pair in itertools.batched(selected_nodes, n=2):
            if len(pair) < 2:
                return "Odd number of gates!", no_update, no_update, no_update
            getattr(g, method_name)(*pair, **method_args)

        ui = f"Applied {method_name} to {selected_nodes}"
    elif method_name in ["X", "Y", "Z", "local_complementation", "H", "S"]:
        for i in selected_nodes:
            getattr(g, method_name)(i, **method_args)

        ui = f"Applied {method_name} to {selected_nodes}"
    elif method_name in ["measure"]:
        outcomes = []
        for i in selected_nodes:
            outcomes.append(getattr(g, method_name)(i, **method_args))

        ui = f"Measured selected nodes {selected_nodes} with outcomes {outcomes}"
    elif method_name in ["fusion_gate"]:
        for pair in itertools.batched(selected_nodes, n=2):
            if len(pair) < 2:
                return "Odd number of qubits!", no_update, no_update, no_update
            getattr(g, method_name)(*pair, **method_args)

        ui = f"Applied {method_name} to {selected_nodes}"

    elif method_name in ["add_node"]:
        getattr(g, method_name)(**method_args)
        positions += [{"x": random.randint(0, 100), "y": random.randint(0, 100)}]
        ui = f"Added node {len(g) - 1}!"
    elif method_name in ["remove_node"]:
        if len(selected_nodes) >= len(g):
            return "Cannot remove last node!", no_update, no_update, no_update

        getattr(g, method_name)(selected_nodes, **method_args)
        ui = f"Removed nodes {selected_nodes}"

        # When a node is removed, the positions need to be updated to adjust
        new_positions = []
        for cyto_index, pos in enumerate(positions):
            if cyto_index not in selected_nodes:
                new_positions.append(pos)

        cyto_data_new = g.to_cytoscape(export_elements=True)
        cyto_data_new = postprocess_cyto_data_elements(cyto_data_new, new_positions)

        log += f"REMOVE_NODE {' '.join(map(str, selected_nodes))}\n"

        return ui, cyto_data_new, repr(g), log

    else:
        raise NotImplementedError(f"Do not know {method_name}")

    cyto_data_new = g.to_cytoscape(export_elements=True)
    cyto_data_new = postprocess_cyto_data_elements(cyto_data_new, positions)

    if method_name == "measure":
        log += f"M{method_args['basis']}"
        if method_args["force"] != -1:
            log += f"[{method_args['force']}]"
        log += f" {' '.join(map(str, selected_nodes))}\n"
        log += f"# OUTCOME {' '.join(map(str, outcomes))}\n"
    elif method_name == "fusion_gate":
        log += f"FUSION_GATE[{method_args['gate_control']}{method_args['gate_target']}] {' '.join(map(str, selected_nodes))}\n"
    elif method_name == "local_complementation":
        log += f"LC {' '.join(map(str, selected_nodes))}\n"
    elif method_name == "add_node":
        log += f"ADD_NODE {len(g) - 1}\n"
    else:
        log += f"{method_name.upper()} {' '.join(map(str, selected_nodes))}\n"

    return ui, cyto_data_new, repr(g), log


def preprocess_cyto_data_elements(
    cyto_data: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    nodes = []
    edges = []
    positions = []
    for item in cyto_data:
        if item.get("data"):
            if item["data"].get("vop"):
                nodes.append(item)  # This is a node
                positions.append(item["position"])
            elif item["data"].get("source"):
                edges.append(item)  # This is an edge
        else:
            raise NotImplementedError("Unknown or no data")
    return {
        "data": [],
        "multigraph": False,
        "elements": {"nodes": nodes, "edges": edges},
    }, positions


def postprocess_cyto_data_elements(
    cyto_data: dict[str, Any], positions: list[dict[str, Any]]
):
    for item, pos in zip(cyto_data, positions):
        if item.get("data"):
            if item["data"].get("vop"):
                item["position"] = pos
        else:
            raise NotImplementedError("Unknown or no data")
    return cyto_data


@callback(
    Output("figure-app", "elements", allow_duplicate=True),
    Output("simulator-representation", "children", allow_duplicate=True),
    Output("move-log", "children", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Input("undo-button", "n_clicks"),
    Input("load-button", "n_clicks"),
    State("load-graph-input", "value"),
    State("move-log", "children"),
    State("figure-app", "elements"),
    prevent_initial_call=True,
)
def load_graph(_undo: int, _load: int, load_graph_input: str, move_log: str, cyto_data):
    """Load or undo from either the move log or a saved text.

    Args:
        _undo (int): whether the undo button was clicked
        _load (int): whether the load button was clicked
        load_graph_input (str): the load graph input
        move_log (str): the move log
    """

    triggered_id = callback_context.triggered_id
    _, positions = preprocess_cyto_data_elements(cyto_data)

    if not load_graph_input and triggered_id == "load-button":
        return no_update, no_update, no_update, "Cannot load empty input!"

    if not move_log and triggered_id == "undo-button":
        return no_update, no_update, no_update, "Empty graph!"

    if triggered_id == "undo-button":
        move_log_list = [move for move in move_log.split("\n") if move]
        if not move_log_list[:-1]:
            return no_update, no_update, no_update, "Cannot undo, empty graph!"

        move_log = "\n".join(move_log_list[:-1])
        g, parsed_log = ClusterState.load_text(move_log, return_log=True)
        
        
    elif triggered_id == "load-button":
        g, parsed_log = ClusterState.load_text(load_graph_input, return_log=True)
        if len(positions) < len(g):
            for i in range(len(g) - len(positions)):
                positions += [{"x": random.randint(0, 100), "y": random.randint(0, 100)}]

    cyto_data_new = g.to_cytoscape(export_elements=True)
    cyto_data_new = postprocess_cyto_data_elements(cyto_data_new, positions)

    return cyto_data_new, repr(g), parsed_log, "Loaded graph!"
