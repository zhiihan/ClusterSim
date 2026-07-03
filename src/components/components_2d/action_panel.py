import itertools
from clustersim.simulator import ClusterState
from textwrap import dedent as d
from dash import dcc, callback, Input, Output, State, no_update, callback_context, html
import dash_bootstrap_components as dbc
from typing import Any
import random

qubit_panel = dbc.Row(
    [
        # Column 1: Select Measurement Basis
        dbc.Col(
            [
                dcc.Markdown("**Select Measurement Basis**", style={"margin": "0 0 4px 0", "fontSize": "0.8rem"}),
                html.Div(
                    [
                        dbc.ButtonGroup(
                            [
                                dbc.Button("MZ", outline=True, color="primary", id="MZ", style={"fontSize": "0.85rem", "padding": "4px 6px"}),
                                dbc.Button("MY", outline=True, color="primary", id="MY", style={"fontSize": "0.85rem", "padding": "4px 6px"}),
                                dbc.Button("MX", outline=True, color="primary", id="MX", style={"fontSize": "0.85rem", "padding": "4px 6px"}),
                            ],
                            className="w-100"
                        ),
                        dbc.Select(
                            id="force-measurement",
                            options=[
                                {"label": "Force 0", "value": 0},
                                {"label": "Force 1", "value": 1},
                                {"label": "Random", "value": -1},
                            ],
                            placeholder="Force 0",
                            value=0,
                            style={"fontSize": "0.85rem", "padding": "4px 8px", "height": "auto"},
                            className="w-100 mt-2"
                        ),
                    ],
                ),
            ],
            width=2
        ),
        # Column 2: Fusion Controls
        dbc.Col(
            [
                dcc.Markdown("**Fusion Controls**", style={"margin": "0 0 4px 0", "fontSize": "0.8rem"}),
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Span("Fusion Type", style={"fontSize": "0.7rem", "fontWeight": "600", "display": "block", "marginBottom": "2px"}),
                                        dbc.Select(
                                            id="fusion-type",
                                            options=[
                                                {"label": "XXZZ", "value": "XXZZ"},
                                                {"label": "XZZX", "value": "XZZX"},
                                                {"label": "XYYZ", "value": "XYYZ"},
                                            ],
                                            value="XXZZ",
                                            style={"fontSize": "0.7rem", "padding": "2px 4px", "height": "auto"},
                                        ),
                                    ],
                                    width=6,
                                    className="pe-1"
                                ),
                                dbc.Col(
                                    [
                                        html.Span("Force Meas.", style={"fontSize": "0.7rem", "fontWeight": "600", "display": "block", "marginBottom": "2px"}),
                                        dbc.Select(
                                            id="fusion-force-measurement",
                                            options=[
                                                {"label": "00", "value": 0},
                                                {"label": "10", "value": 1},
                                                {"label": "01", "value": 2},
                                                {"label": "11", "value": 3},
                                                {"label": "Random", "value": -1},
                                            ],
                                            placeholder="00",
                                            value=0,
                                            style={"fontSize": "0.7rem", "padding": "2px 4px", "height": "auto"},
                                        ),
                                    ],
                                    width=6,
                                    className="ps-1"
                                ),
                            ],
                            className="mb-2 g-0"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Span("Fusion Mode", style={"fontSize": "0.7rem", "fontWeight": "600", "display": "block", "marginBottom": "2px"}),
                                        dbc.Select(
                                            id="fusion-mode",
                                            options=[
                                                {"label": "Success", "value": "success"},
                                                {"label": "Failure", "value": "failure"},
                                                {"label": "Random", "value": "random"},
                                            ],
                                            value="success",
                                            style={"fontSize": "0.7rem", "padding": "2px 4px", "height": "auto"},
                                        ),
                                    ],
                                    width=6,
                                    className="pe-1"
                                ),
                                dbc.Col(
                                    [
                                        html.Span(" ", style={"fontSize": "0.7rem", "display": "block", "marginBottom": "2px"}),
                                        dbc.Button("Fusion", outline=True, color="primary", id="fusion-gate", style={"fontSize": "0.7rem", "padding": "2px 6px", "height": "auto"}, className="w-100"),
                                    ],
                                    width=6,
                                    className="ps-1 d-flex flex-column justify-content-end"
                                ),
                            ],
                            className="g-0"
                        ),
                    ],
                ),
            ],
            width=3
        ),
        # Column 3: Apply Clifford Gates
        dbc.Col(
            [
                dcc.Markdown("**Apply Clifford Gates**", style={"margin": "0 0 4px 0", "fontSize": "0.8rem"}),
                html.Div(
                    [
                        dbc.ButtonGroup(
                            [
                                dbc.Button("X", outline=True, color="primary", id="X", style={"fontSize": "0.85rem", "padding": "4px 6px"}),
                                dbc.Button("Y", outline=True, color="primary", id="Y", style={"fontSize": "0.85rem", "padding": "4px 6px"}),
                                dbc.Button("Z", outline=True, color="primary", id="Z", style={"fontSize": "0.85rem", "padding": "4px 6px"}),
                                dbc.Button("H", outline=True, color="primary", id="H", style={"fontSize": "0.85rem", "padding": "4px 6px"}),
                            ],
                            className="w-100 mb-2"
                        ),
                        dbc.ButtonGroup(
                            [
                                dbc.Button("S", outline=True, color="primary", id="S", style={"fontSize": "0.85rem", "padding": "4px 6px"}),
                                dbc.Button("S†", outline=True, color="primary", id="S_DAG", style={"fontSize": "0.85rem", "padding": "4px 6px"}),
                                dbc.Button("CX", outline=True, color="primary", id="CX", style={"fontSize": "0.85rem", "padding": "4px 6px"}),
                                dbc.Button("CZ", outline=True, color="primary", id="CZ", style={"fontSize": "0.85rem", "padding": "4px 6px"}),
                            ],
                            className="w-100"
                        ),
                    ]
                ),
            ],
            width=3
        ),
        # Column 4: Graph operations
        dbc.Col(
            [
                dcc.Markdown("**Graph operations**", style={"margin": "0 0 4px 0", "fontSize": "0.8rem"}),
                html.Div(
                    [
                        dbc.ButtonGroup(
                            [
                                dbc.Button("Add Node", outline=True, color="primary", id="add-node", style={"fontSize": "0.8rem", "padding": "4px 6px"}),
                                dbc.Button("Del Node", outline=True, color="primary", id="remove-node", style={"fontSize": "0.8rem", "padding": "4px 6px"}),
                            ],
                            className="w-100 mb-2"
                        ),
                        dbc.ButtonGroup(
                            [
                                dbc.Button("Toggle", outline=True, color="primary", id="toggle-edge", style={"fontSize": "0.8rem", "padding": "4px 6px"}),
                                dbc.Button("Del Edge", outline=True, color="primary", id="remove-edge", style={"fontSize": "0.8rem", "padding": "4px 6px"}),
                            ],
                            className="w-100 mb-2"
                        ),
                        dbc.ButtonGroup(
                            [
                                dbc.Button("LC", outline=True, color="primary", id="LC", style={"fontSize": "0.8rem", "padding": "4px 6px"}),
                                dbc.Button("LC Rewrite", outline=True, color="primary", id="LCR", style={"fontSize": "0.8rem", "padding": "4px 6px"}),
                                dbc.Button("Copy", outline=True, color="primary", id="duplicate", style={"fontSize": "0.8rem", "padding": "4px 6px"}),
                            ],
                            className="w-100"
                        ),
                        dbc.Button("Add Edge", outline=True, color="primary", id="add-edge", style={"display": "none"}),
                    ]
                ),
            ],
            width=3
        ),
        # Column 5: Advanced Options (Gear icon)
        dbc.Col(
            [
                html.Div(style={"height": "20px"}),
                html.Div(
                    [
                        dbc.Button("⚙", id="open-modal-btn", outline=True, color="secondary", style={"fontSize": "1.2rem", "padding": "2px 8px"}, title="Advanced Options"),
                    ],
                    className="w-100 d-flex justify-content-center mt-1"
                ),
            ],
            width=1
        ),
    ],
    className="flex-nowrap"
)


button_operations = {
    # measure buttons with basis
    "MX": ("measure", {"force": 0, "basis": "X"}),
    "MY": ("measure", {"force": 0, "basis": "Y"}),
    "MZ": ("measure", {"force": 0, "basis": "Z"}),
    "fusion-gate": ("fusion_gate", {"force": 0, "mode": "success"}),
    # simple operations with no extra args
    "LCR": ("local_complementation_rewrite", {}),
    "LC": ("local_complementation", {}),
    "X": ("X", {}),
    "Y": ("Y", {}),
    "Z": ("Z", {}),
    "H": ("H", {}),
    "S": ("S", {}),
    "S_DAG": ("S_DAG", {}),
    "CX": ("CX", {}),
    "CZ": ("CZ", {}),
    "add-edge": ("add_edge", {}),
    "remove-edge": ("remove_edge", {}),
    "toggle-edge": ("toggle_edge", {}),
    "add-node": ("add_node", {"vop": "IA"}),
    "remove-node": ("remove_node", {}),
    "duplicate": ("duplicate", {}),
}


@callback(
    Output("ui", "children"),
    Output("figure-app", "elements", allow_duplicate=True),
    Output("simulator-representation", "children"),
    Output("move-log", "children"),
    Output("history-store", "data", allow_duplicate=True),
    *[Input(btn, "n_clicks") for btn in button_operations.keys()],
    Input("backspace-btn", "n_clicks"),
    Input("key-n-btn", "n_clicks"),
    Input("key-e-btn", "n_clicks"),
    Input("key-shift-e-btn", "n_clicks"),
    Input("key-h-btn", "n_clicks"),
    Input("key-w-btn", "n_clicks"),
    Input("ctrl-v-btn", "n_clicks"),
    State("clipboard-store", "data"),
    State("history-store", "data"),
    State("fusion-mode", "value"),
    State("fusion-force-measurement", "value"),
    State("fusion-type", "value"),
    State("force-measurement", "value"),
    State("move-log", "children"),
    State("figure-app", "selectedNodeData"),
    State("figure-app", "selectedEdgeData"),
    State("figure-app", "elements"),
    prevent_initial_call=True,
)
def handle_buttons(*args):
    kwargs = {}

    log = args[-4]

    # The last three args are selectedNodeData, selectedEdgeData, and elements
    selected_node_data = args[-3]
    selected_edge_data = args[-2]
    cyto_data = args[-1]

    history_data = args[-9]
    if not history_data or not isinstance(history_data, dict):
        history_data = {"undo_stack": [], "redo_stack": []}
    else:
        history_data = {
            "undo_stack": list(history_data.get("undo_stack", [])),
            "redo_stack": list(history_data.get("redo_stack", [])),
        }

    # Record the current state in history before performing the action
    _, current_positions = preprocess_cyto_data_elements(cyto_data)
    history_data["undo_stack"].append({"move_log": log, "positions": current_positions})
    history_data["redo_stack"] = []

    # Determine which button triggered the callback
    triggered_id = callback_context.triggered_id
    if triggered_id == "backspace-btn":
        triggered_id = "remove-node"
    elif triggered_id == "key-n-btn":
        triggered_id = "add-node"
    elif triggered_id == "key-e-btn":
        triggered_id = "toggle-edge"
    elif triggered_id == "key-shift-e-btn":
        triggered_id = "remove-edge"
    elif triggered_id == "key-h-btn":
        triggered_id = "H"
    elif triggered_id == "key-w-btn":
        triggered_id = "LC"

    if triggered_id == "ctrl-v-btn":
        clipboard_data = args[-10]
        if not clipboard_data:
            return (
                "Clipboard is empty!",
                no_update,
                no_update,
                no_update,
                no_update,
            )
        selected_nodes = clipboard_data
        method_name = "duplicate"
        method_args = {}
    else:
        if not triggered_id or triggered_id not in button_operations:
            raise NotImplementedError

        method_name, method_args = button_operations[triggered_id]
        if method_name == "fusion_gate":
            method_args["fusion_type"] = args[-6]
            method_args["force"] = int(args[-7])
            method_args["mode"] = args[-8]

        elif method_name == "measure":
            method_args["force"] = int(args[-5])

        selected_nodes = [i["value"] for i in selected_node_data]

        # If no nodes selected but an edge is selected, derive nodes from the edge
        # for edge operations (add/remove/toggle)
        if not selected_nodes and method_name in [
            "add_edge",
            "remove_edge",
            "toggle_edge",
        ]:
            if selected_edge_data:
                edge_nodes: list[int] = []
                for edge in selected_edge_data:
                    src = edge.get("source")
                    tgt = edge.get("target")
                    if src is not None:
                        edge_nodes.append(int(src))
                    if tgt is not None:
                        edge_nodes.append(int(tgt))
                selected_nodes = list(
                    dict.fromkeys(edge_nodes)
                )  # deduplicate, preserve order

    if not selected_nodes and method_name != "add_node":
        return no_update, no_update, no_update, no_update, no_update

    return apply_operation_wrapper(
        method_name, selected_nodes, cyto_data, log, method_args, history_data, **kwargs
    )


def apply_operation_wrapper(
    method_name: str,
    selected_nodes: list[int],
    cyto_data,
    log,
    method_args,
    history_data,
    **kwargs,
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
                return "Odd number of gates!", no_update, no_update, no_update, no_update
            getattr(g, method_name)(*pair, **method_args)

        ui = f"Applied {method_name} to {selected_nodes}"
    elif method_name in [
        "X",
        "Y",
        "Z",
        "local_complementation",
        "local_complementation_rewrite",
        "H",
        "S",
    ]:
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
                return "Odd number of qubits!", no_update, no_update, no_update, no_update
            getattr(g, method_name)(*pair, **method_args)

        ui = f"Applied {method_name} to {selected_nodes}"

    elif method_name in ["add_node"]:
        getattr(g, method_name)(**method_args)
        new_node_id = len(g) - 1

        selected_positions = []
        if selected_nodes:
            for parent_id in selected_nodes:
                parent_pos = next(
                    (
                        item["position"]
                        for item in cyto_data["elements"]["nodes"]
                        if item.get("data") and item["data"].get("value") == parent_id
                    ),
                    None,
                )
                if parent_pos:
                    selected_positions.append(parent_pos)

        if selected_positions:
            avg_x = sum(p["x"] for p in selected_positions) / len(selected_positions)
            avg_y = sum(p["y"] for p in selected_positions) / len(selected_positions)
            new_pos = {"x": avg_x + 50, "y": avg_y + 50}
        else:
            new_pos = {"x": random.randint(0, 100), "y": random.randint(0, 100)}

        positions.append(new_pos)

        # Connect the new node to every selected node
        for neighbour in selected_nodes:
            g.add_edge(new_node_id, neighbour)
        if selected_nodes:
            ui = f"Added node {new_node_id} with edges to {selected_nodes}!"
        else:
            ui = f"Added node {new_node_id}!"
    elif method_name in ["remove_node"]:
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

        return ui, cyto_data_new, repr(g), log, history_data
    elif method_name == "duplicate":
        selected_nodes.sort()
        for parent_id in selected_nodes:
            parent_pos = next(
                (
                    item["position"]
                    for item in cyto_data["elements"]["nodes"]
                    if item.get("data") and item["data"].get("value") == parent_id
                ),
                {"x": 0, "y": 0},
            )
            positions.append({"x": parent_pos["x"] + 50, "y": parent_pos["y"] + 50})

        g = getattr(g, method_name)(selected_nodes, **method_args)
        ui = f"Duplicated nodes {selected_nodes}"
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
        log += f"FUSION_GATE[{method_args['fusion_type']}] {' '.join(map(str, selected_nodes))}\n"
    elif method_name == "local_complementation":
        log += f"LC {' '.join(map(str, selected_nodes))}\n"
    elif method_name == "local_complementation_rewrite":
        log += f"LCR {' '.join(map(str, selected_nodes))}\n"
    elif method_name == "add_node":
        new_node_id = len(g) - 1
        log += f"ADD_NODE {new_node_id}\n"
        for neighbour in selected_nodes:
            log += f"ADD_EDGE {new_node_id} {neighbour}\n"
    else:
        log += f"{method_name.upper()} {' '.join(map(str, selected_nodes))}\n"

    return ui, cyto_data_new, repr(g), log, history_data


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
    Output("history-store", "data", allow_duplicate=True),
    Output("load-graph-input", "valid", allow_duplicate=True),
    Output("load-graph-input", "invalid", allow_duplicate=True),
    Output("load-feedback-invalid", "children", allow_duplicate=True),
    Input("undo-button", "n_clicks"),
    Input("ctrl-z-btn", "n_clicks"),
    Input("redo-button", "n_clicks"),
    Input("ctrl-y-btn", "n_clicks"),
    Input("load-button", "n_clicks"),
    State("load-graph-input", "value"),
    State("move-log", "children"),
    State("figure-app", "elements"),
    State("history-store", "data"),
    prevent_initial_call=True,
)
def load_graph(
    _undo,
    _ctrl_z,
    _redo,
    _ctrl_y,
    _load,
    load_graph_input,
    move_log,
    cyto_data,
    history_data,
):
    """Load, undo, or redo from move log or saved text."""

    triggered_id = callback_context.triggered_id
    if triggered_id == "ctrl-z-btn":
        triggered_id = "undo-button"
    elif triggered_id == "ctrl-y-btn":
        triggered_id = "redo-button"

    if not history_data or not isinstance(history_data, dict):
        history_data = {"undo_stack": [], "redo_stack": []}
    else:
        history_data = {
            "undo_stack": list(history_data.get("undo_stack", [])),
            "redo_stack": list(history_data.get("redo_stack", [])),
        }

    _, positions = preprocess_cyto_data_elements(cyto_data)

    valid = False
    invalid = False
    feedback = ""

    if not load_graph_input and triggered_id == "load-button":
        return no_update, no_update, no_update, "Cannot load empty input!", no_update, False, True, "Cannot load empty input!"

    if triggered_id == "undo-button":
        if not history_data["undo_stack"]:
            return no_update, no_update, no_update, "Nothing to undo!", no_update, False, False, ""

        # Pop the state to restore
        prev_state = history_data["undo_stack"].pop()

        # Save the current state to the redo stack
        _, current_positions = preprocess_cyto_data_elements(cyto_data)
        history_data["redo_stack"].append(
            {"move_log": move_log, "positions": current_positions}
        )

        move_log = prev_state["move_log"]
        positions = prev_state["positions"]
        g, parsed_log = ClusterState.from_string(move_log, return_log=True)

    elif triggered_id == "redo-button":
        if not history_data["redo_stack"]:
            return no_update, no_update, no_update, "Nothing to redo!", no_update, False, False, ""

        # Pop the state to restore
        next_state = history_data["redo_stack"].pop()

        # Save the current state to the undo stack
        _, current_positions = preprocess_cyto_data_elements(cyto_data)
        history_data["undo_stack"].append(
            {"move_log": move_log, "positions": current_positions}
        )

        move_log = next_state["move_log"]
        positions = next_state["positions"]
        g, parsed_log = ClusterState.from_string(move_log, return_log=True)

    elif triggered_id == "load-button":
        try:
            g, parsed_log = ClusterState.from_string(load_graph_input, return_log=True)
            valid = True
            invalid = False
        except Exception as e:
            err_msg = str(e)
            return no_update, no_update, no_update, f"Failed to load: {err_msg}", no_update, False, True, err_msg

        if len(positions) < len(g):
            for i in range(len(g) - len(positions)):
                positions += [
                    {"x": random.randint(0, 300), "y": random.randint(0, 300)}
                ]
        positions = positions[: len(g)]
        move_log = parsed_log
        # Clear history on new graph load
        history_data = {"undo_stack": [], "redo_stack": []}

    cyto_data_new = g.to_cytoscape(export_elements=True)
    cyto_data_new = postprocess_cyto_data_elements(cyto_data_new, positions)

    return cyto_data_new, repr(g), move_log, "Action completed!", history_data, valid, invalid, feedback


@callback(
    Output("download-json", "data"),
    Input("btn-get-json", "n_clicks"),
    State("figure-app", "elements"),
    prevent_initial_call=True,
)
def export_graph_to_json(n_clicks, cyto_data):
    if not n_clicks:
        return no_update
    cyto_data, positions = preprocess_cyto_data_elements(cyto_data)
    g = ClusterState.from_cytoscape(cyto_data)
    return dcc.send_string(g.to_json(), "cytoscape_graph.json")


@callback(
    Output("clipboard-store", "data"),
    Input("ctrl-c-btn", "n_clicks"),
    State("figure-app", "selectedNodeData"),
    prevent_initial_call=True,
)
def copy_nodes(n_clicks, selected_node_data):
    if not selected_node_data:
        return []
    return [node["value"] for node in selected_node_data]
