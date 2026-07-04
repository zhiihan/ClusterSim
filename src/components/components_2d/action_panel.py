import itertools
from clustersim.simulator import ClusterState, FrontendClusterState
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
                dcc.Markdown(
                    "**Select Measurement Basis**",
                    style={"margin": "0 0 4px 0", "fontSize": "0.8rem"},
                ),
                html.Div(
                    [
                        dbc.ButtonGroup(
                            [
                                dbc.Button(
                                    "MZ",
                                    outline=True,
                                    color="primary",
                                    id="MZ",
                                    style={"fontSize": "0.85rem", "padding": "4px 6px"},
                                ),
                                dbc.Button(
                                    "MY",
                                    outline=True,
                                    color="primary",
                                    id="MY",
                                    style={"fontSize": "0.85rem", "padding": "4px 6px"},
                                ),
                                dbc.Button(
                                    "MX",
                                    outline=True,
                                    color="primary",
                                    id="MX",
                                    style={"fontSize": "0.85rem", "padding": "4px 6px"},
                                ),
                            ],
                            className="w-100",
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
                            style={
                                "fontSize": "0.85rem",
                                "padding": "4px 8px",
                                "height": "auto",
                            },
                            className="w-100 mt-2",
                        ),
                    ],
                ),
            ],
            width=2,
        ),
        # Column 2: Fusion Controls
        dbc.Col(
            [
                dcc.Markdown(
                    "**Fusion Controls**",
                    style={"margin": "0 0 4px 0", "fontSize": "0.8rem"},
                ),
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Span(
                                            "Force Meas.",
                                            style={
                                                "fontSize": "0.85rem",
                                                "fontWeight": "600",
                                                "display": "block",
                                                "marginBottom": "2px",
                                            },
                                        ),
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
                                            style={
                                                "fontSize": "0.85rem",
                                                "padding": "2px 4px",
                                                "height": "auto",
                                            },
                                        ),
                                    ],
                                    width=6,
                                    className="pe-1",
                                ),
                                dbc.Col(
                                    [
                                        html.Span(
                                            "Fusion Mode",
                                            style={
                                                "fontSize": "0.85rem",
                                                "fontWeight": "600",
                                                "display": "block",
                                                "marginBottom": "2px",
                                            },
                                        ),
                                        dbc.Select(
                                            id="fusion-mode",
                                            options=[
                                                {
                                                    "label": "Success",
                                                    "value": "success",
                                                },
                                                {
                                                    "label": "Failure",
                                                    "value": "failure",
                                                },
                                                {"label": "Random", "value": "random"},
                                            ],
                                            value="success",
                                            style={
                                                "fontSize": "0.85rem",
                                                "padding": "2px 4px",
                                                "height": "auto",
                                            },
                                        ),
                                    ],
                                    width=6,
                                    className="ps-1",
                                ),
                            ],
                            className="mb-2 g-0",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.ButtonGroup(
                                            [
                                                dbc.Button(
                                                    "XXZZ",
                                                    outline=True,
                                                    color="primary",
                                                    id="fusion-gate-xxzz",
                                                    style={
                                                        "fontSize": "0.8rem",
                                                        "padding": "4px 2px",
                                                    },
                                                ),
                                                dbc.Button(
                                                    "XZZX",
                                                    outline=True,
                                                    color="primary",
                                                    id="fusion-gate-xzzx",
                                                    style={
                                                        "fontSize": "0.8rem",
                                                        "padding": "4px 2px",
                                                    },
                                                ),
                                                dbc.Button(
                                                    "XYYZ",
                                                    outline=True,
                                                    color="primary",
                                                    id="fusion-gate-xyyz",
                                                    style={
                                                        "fontSize": "0.8rem",
                                                        "padding": "4px 2px",
                                                    },
                                                ),
                                            ],
                                            className="w-100",
                                        ),
                                    ],
                                    width=12,
                                ),
                            ],
                            className="g-0",
                        ),
                    ],
                ),
            ],
            width=3,
        ),
        # Column 3: Apply Clifford Gates
        dbc.Col(
            [
                dcc.Markdown(
                    "**Apply Clifford Gates**",
                    style={"margin": "0 0 4px 0", "fontSize": "0.8rem"},
                ),
                html.Div(
                    [
                        dbc.ButtonGroup(
                            [
                                dbc.Button(
                                    "X",
                                    outline=True,
                                    color="primary",
                                    id="X",
                                    style={"fontSize": "0.85rem", "padding": "4px 6px"},
                                ),
                                dbc.Button(
                                    "Y",
                                    outline=True,
                                    color="primary",
                                    id="Y",
                                    style={"fontSize": "0.85rem", "padding": "4px 6px"},
                                ),
                                dbc.Button(
                                    "Z",
                                    outline=True,
                                    color="primary",
                                    id="Z",
                                    style={"fontSize": "0.85rem", "padding": "4px 6px"},
                                ),
                                dbc.Button(
                                    "H",
                                    outline=True,
                                    color="primary",
                                    id="H",
                                    style={"fontSize": "0.85rem", "padding": "4px 6px"},
                                ),
                            ],
                            className="w-100 mb-2",
                        ),
                        dbc.ButtonGroup(
                            [
                                dbc.Button(
                                    "S",
                                    outline=True,
                                    color="primary",
                                    id="S",
                                    style={"fontSize": "0.85rem", "padding": "4px 6px"},
                                ),
                                dbc.Button(
                                    "S†",
                                    outline=True,
                                    color="primary",
                                    id="S_DAG",
                                    style={"fontSize": "0.85rem", "padding": "4px 6px"},
                                ),
                                dbc.Button(
                                    "CX",
                                    outline=True,
                                    color="primary",
                                    id="CX",
                                    style={"fontSize": "0.85rem", "padding": "4px 6px"},
                                ),
                                dbc.Button(
                                    "CZ",
                                    outline=True,
                                    color="primary",
                                    id="CZ",
                                    style={"fontSize": "0.85rem", "padding": "4px 6px"},
                                ),
                            ],
                            className="w-100",
                        ),
                    ]
                ),
            ],
            width=3,
        ),
        # Column 4: Graph operations
        dbc.Col(
            [
                dcc.Markdown(
                    "**Graph operations**",
                    style={"margin": "0 0 4px 0", "fontSize": "0.8rem"},
                ),
                html.Div(
                    [
                        dbc.ButtonGroup(
                            [
                                dbc.Button(
                                    "Add Node",
                                    outline=True,
                                    color="primary",
                                    id="add-node",
                                    style={"fontSize": "0.8rem", "padding": "4px 6px"},
                                ),
                                dbc.Button(
                                    "Del Node",
                                    outline=True,
                                    color="primary",
                                    id="remove-node",
                                    style={"fontSize": "0.8rem", "padding": "4px 6px"},
                                ),
                            ],
                            className="w-100 mb-2",
                        ),
                        dbc.ButtonGroup(
                            [
                                dbc.Button(
                                    "Toggle",
                                    outline=True,
                                    color="primary",
                                    id="toggle-edge",
                                    style={"fontSize": "0.8rem", "padding": "4px 6px"},
                                ),
                                dbc.Button(
                                    "Del Edge",
                                    outline=True,
                                    color="primary",
                                    id="remove-edge",
                                    style={"fontSize": "0.8rem", "padding": "4px 6px"},
                                ),
                            ],
                            className="w-100 mb-2",
                        ),
                        dbc.ButtonGroup(
                            [
                                dbc.Button(
                                    "LC",
                                    outline=True,
                                    color="primary",
                                    id="LC",
                                    style={"fontSize": "0.8rem", "padding": "4px 6px"},
                                ),
                                dbc.Button(
                                    "LC Rewrite",
                                    outline=True,
                                    color="primary",
                                    id="LCR",
                                    style={"fontSize": "0.8rem", "padding": "4px 6px"},
                                ),
                                dbc.Button(
                                    "Copy",
                                    outline=True,
                                    color="primary",
                                    id="duplicate",
                                    style={"fontSize": "0.8rem", "padding": "4px 6px"},
                                ),
                            ],
                            className="w-100",
                        ),
                        dbc.Button(
                            "Add Edge",
                            outline=True,
                            color="primary",
                            id="add-edge",
                            style={"display": "none"},
                        ),
                    ]
                ),
            ],
            width=3,
        ),
        # Column 5: Advanced Options (Gear and Hamburger icon)
        dbc.Col(
            [
                html.Div(
                    [
                        dbc.Button(
                            "☰",
                            id="open-offcanvas-btn",
                            outline=True,
                            color="secondary",
                            style={
                                "fontSize": "1.2rem",
                                "padding": "2px 8px",
                                "marginBottom": "4px",
                            },
                            title="Plot Customization Options",
                        ),
                        dbc.Button(
                            "⚙",
                            id="open-modal-btn",
                            outline=True,
                            color="secondary",
                            style={"fontSize": "1.2rem", "padding": "2px 8px"},
                            title="Advanced Options",
                        ),
                    ],
                    className="w-100 d-flex flex-column align-items-center mt-1",
                ),
            ],
            width=1,
        ),
        dbc.Tooltip(
            "Copy (Ctrl+C) / Paste (Ctrl+V)", target="duplicate", placement="top"
        ),
        dbc.Tooltip("Delete Node (Backspace)", target="remove-node", placement="top"),
        dbc.Tooltip("Add Node (N)", target="add-node", placement="top"),
        dbc.Tooltip("Toggle Edge (E)", target="toggle-edge", placement="top"),
        dbc.Tooltip("Delete Edge (Shift+E)", target="remove-edge", placement="top"),
        dbc.Tooltip("Hadamard Gate (H)", target="H", placement="top"),
        dbc.Tooltip("Local Complementation (W)", target="LC", placement="top"),
        dbc.Tooltip("Fusion Gate XXZZ", target="fusion-gate-xxzz", placement="top"),
        dbc.Tooltip("Fusion Gate XZZX", target="fusion-gate-xzzx", placement="top"),
        dbc.Tooltip("Fusion Gate XYYZ", target="fusion-gate-xyyz", placement="top"),
        dbc.Tooltip("Pauli X", target="X", placement="top"),
        dbc.Tooltip("Pauli Y", target="Y", placement="top"),
        dbc.Tooltip("Pauli Z", target="Z", placement="top"),
        dbc.Tooltip("Pauli S", target="S", placement="top"),
        dbc.Tooltip("Pauli S Dagger", target="S_DAG", placement="top"),
        dbc.Tooltip("Controlled X", target="CX", placement="top"),
        dbc.Tooltip("Controlled Z", target="CZ", placement="top"),
        dbc.Tooltip("Rewrite", target="LCR", placement="top"),
        dbc.Tooltip("Measure X", target="MX", placement="top"),
        dbc.Tooltip("Measure Y", target="MY", placement="top"),
        dbc.Tooltip("Measure Z", target="MZ", placement="top"),
    ],
    className="flex-nowrap",
)


button_operations = {
    # measure buttons with basis
    "MX": ("measure", {"force": 0, "basis": "X"}),
    "MY": ("measure", {"force": 0, "basis": "Y"}),
    "MZ": ("measure", {"force": 0, "basis": "Z"}),
    "fusion-gate-xxzz": (
        "fusion_gate",
        {"force": 0, "mode": "success", "fusion_type": "XXZZ"},
    ),
    "fusion-gate-xzzx": (
        "fusion_gate",
        {"force": 0, "mode": "success", "fusion_type": "XZZX"},
    ),
    "fusion-gate-xyyz": (
        "fusion_gate",
        {"force": 0, "mode": "success", "fusion_type": "XYYZ"},
    ),
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

    state = FrontendClusterState.from_cytoscape_elements(cyto_data)

    history_data = args[-8]
    if not history_data or not isinstance(history_data, dict):
        history_data = {"undo_stack": [], "redo_stack": []}
    else:
        history_data = {
            "undo_stack": list(history_data.get("undo_stack", [])),
            "redo_stack": list(history_data.get("redo_stack", [])),
        }

    # Record the current elements state in history for Undo/Redo
    history_data["undo_stack"].append(
        {"action_type": "quantum_operation", "move_log": log, "elements": cyto_data}
    )
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
        clipboard_data = args[-9]
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

        method_name, base_args = button_operations[triggered_id]
        method_args = base_args.copy()
        if method_name == "fusion_gate":
            method_args["force"] = int(args[-6])
            method_args["mode"] = args[-7]

        elif method_name == "measure":
            method_args["force"] = int(args[-5])

        selected_nodes = []
        for i in selected_node_data:
            elem_id = str(i.get("id")) if i.get("id") is not None else ""
            if elem_id and elem_id in state.groups:
                selected_nodes.append(elem_id)
            elif elem_id and elem_id.isdigit():
                selected_nodes.append(elem_id)
            elif i.get("value") is not None:
                selected_nodes.append(i["value"])

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
    selected_nodes: list,
    cyto_data,
    log,
    method_args,
    history_data,
    **kwargs,
):
    """Take the buttons for all the callbacks and apply the corresponding method in FrontendClusterState.

    Args:
        operation_name (str): ClusterState.method_name
        selected_node_data (_type_): selected_node_data
        cyto_data (_type_): cyto_data directly from figure-app.elements

    Returns:
        _type_: processed cyto_data for use in figure-app.elements
    """
    state = FrontendClusterState.from_cytoscape_elements(cyto_data)
    expanded_qubits = state.expand_selection(selected_nodes)
    display_nodes = list(expanded_qubits)

    if method_name in ["add_edge", "remove_edge", "toggle_edge"]:
        for pair in itertools.combinations(expanded_qubits, 2):
            getattr(state, method_name)(*pair, **method_args)
        ui = f"Applied {method_name} to {display_nodes}"
    elif method_name in ["CX", "CZ"]:
        for pair in itertools.batched(expanded_qubits, n=2):
            if len(pair) < 2:
                return (
                    "Odd number of gates!",
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                )
            getattr(state, method_name)(*pair, **method_args)

        ui = f"Applied {method_name} to {display_nodes}"
    elif method_name in [
        "X",
        "Y",
        "Z",
        "local_complementation",
        "local_complementation_rewrite",
        "H",
        "S",
        "S_DAG",
    ]:
        for i in expanded_qubits:
            getattr(state, method_name)(i, **method_args)

        ui = f"Applied {method_name} to {display_nodes}"
    elif method_name in ["measure"]:
        outcomes = []
        for i in expanded_qubits:
            outcomes.append(getattr(state, method_name)(i, **method_args))

        ui = f"Measured selected nodes {display_nodes} with outcomes {outcomes}"
    elif method_name in ["fusion_gate"]:
        for pair in itertools.batched(expanded_qubits, n=2):
            if len(pair) < 2:
                return (
                    "Odd number of qubits!",
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                )
            getattr(state, method_name)(*pair, **method_args)

        ui = f"Applied {method_name} to {display_nodes}"

    elif method_name in ["add_node"]:
        selected_positions = []
        resolved_selected = []
        if expanded_qubits:
            for q in expanded_qubits:
                idx = state.resolve_qubit(q)
                if idx is not None and idx < len(state):
                    resolved_selected.append(idx)
                    selected_positions.append(state.get_position(idx))

        if selected_positions:
            avg_x = sum(p["x"] for p in selected_positions) / len(selected_positions)
            avg_y = sum(p["y"] for p in selected_positions) / len(selected_positions)
            new_pos = {"x": avg_x + 50, "y": avg_y + 50}
        else:
            new_pos = {
                "x": float(random.randint(0, 100)),
                "y": float(random.randint(0, 100)),
            }

        new_node_id = state.add_node(position=new_pos, **method_args)

        # Connect the new node to every selected node
        for neighbour in resolved_selected:
            if neighbour < new_node_id:
                state.add_edge(new_node_id, neighbour)
        if expanded_qubits:
            ui = f"Added node {new_node_id} with edges to {display_nodes}!"
        else:
            ui = f"Added node {new_node_id}!"
    elif method_name in ["remove_node"]:
        groups_to_remove = [q for q in selected_nodes if str(q) in state.groups]

        for g_id in groups_to_remove:
            state.remove_group(str(g_id), unassign_children=True)

        if expanded_qubits:
            state.remove_node(expanded_qubits)

        ui = f"Removed nodes {display_nodes}"
        log += f"REMOVE_NODE {' '.join(map(str, expanded_qubits))}\n"
        cyto_data_new = state.to_cytoscape_elements()
        return ui, cyto_data_new, repr(state.cluster_state), log, history_data
    elif method_name == "duplicate":
        resolved_selected = list(expanded_qubits)
        resolved_selected.sort()
        for q in resolved_selected:
            if q < len(state):
                pos = state.get_position(q)
                state.add_node(position={"x": pos["x"] + 50, "y": pos["y"] + 50})

        state.cluster_state = state.cluster_state.duplicate(resolved_selected)
        ui = f"Duplicated nodes {display_nodes}"
    else:
        raise NotImplementedError(f"Do not know {method_name}")

    cyto_data_new = state.to_cytoscape_elements()

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
        new_node_id = len(state) - 1
        log += f"ADD_NODE {new_node_id}\n"
        for neighbour in selected_nodes:
            log += f"ADD_EDGE {new_node_id} {neighbour}\n"
    else:
        log += f"{method_name.upper()} {' '.join(map(str, selected_nodes))}\n"

    return ui, cyto_data_new, repr(state.cluster_state), log, history_data


def preprocess_cyto_data_elements(
    cyto_data: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    nodes = []
    edges = []
    positions = []
    for item in cyto_data:
        if item.get("data"):
            if item["data"].get("vop"):
                nodes.append(item)  # This is a quantum qubit node
                positions.append(item.get("position", {"x": 0.0, "y": 0.0}))
            elif item["data"].get("source"):
                edges.append(item)  # This is an edge
            else:
                # Compound group container node (no vop, no source)
                pass
        else:
            pass
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
        return (
            no_update,
            no_update,
            no_update,
            "Cannot load empty input!",
            no_update,
            False,
            True,
            "Cannot load empty input!",
        )

    if triggered_id == "undo-button":
        if not history_data["undo_stack"]:
            return (
                no_update,
                no_update,
                no_update,
                "Nothing to undo!",
                no_update,
                False,
                False,
                "",
            )

        # Pop the state to restore
        prev_state = history_data["undo_stack"].pop()

        # Save the current state to the redo stack
        history_data["redo_stack"].append(
            {
                "action_type": prev_state.get("action_type", "quantum_operation"),
                "move_log": move_log,
                "elements": cyto_data,
            }
        )

        action_type = prev_state.get("action_type", "quantum_operation")
        move_log = prev_state.get("move_log", move_log)
        cyto_data_new = prev_state["elements"]

        if action_type == "frontend_customization":
            # Undoing a plot customization (group creation/removal, label, shape):
            # Restore FrontendClusterState elements without resetting simulator state or move_log
            state = FrontendClusterState.from_cytoscape_elements(cyto_data_new)
            return (
                cyto_data_new,
                repr(state.cluster_state),
                move_log,
                "Undid plot customization!",
                history_data,
                valid,
                invalid,
                feedback,
            )
        else:
            # Undoing a quantum operation: restore quantum state while preserving active groups and plot customizations
            curr_state = FrontendClusterState.from_cytoscape_elements(cyto_data)
            state = FrontendClusterState.from_cytoscape_elements(cyto_data_new)

            # Preserve active groups from curr_state if present
            for g_id, g_info in curr_state.groups.items():
                if g_id not in state.groups:
                    state.add_group(
                        g_id, label=g_info.get("label"), style=g_info.get("style")
                    )

            cyto_data_new = state.to_cytoscape_elements()

            return (
                cyto_data_new,
                repr(state.cluster_state),
                move_log,
                "Undid operation!",
                history_data,
                valid,
                invalid,
                feedback,
            )

    elif triggered_id == "redo-button":
        if not history_data["redo_stack"]:
            return (
                no_update,
                no_update,
                no_update,
                "Nothing to redo!",
                no_update,
                False,
                False,
                "",
            )

        # Pop the state to restore
        next_state = history_data["redo_stack"].pop()

        # Save the current state to the undo stack
        history_data["undo_stack"].append(
            {
                "action_type": next_state.get("action_type", "quantum_operation"),
                "move_log": move_log,
                "elements": cyto_data,
            }
        )

        action_type = next_state.get("action_type", "quantum_operation")
        move_log = next_state.get("move_log", move_log)
        cyto_data_new = next_state["elements"]

        state = FrontendClusterState.from_cytoscape_elements(cyto_data_new)
        return (
            cyto_data_new,
            repr(state.cluster_state),
            move_log,
            "Action completed!",
            history_data,
            valid,
            invalid,
            feedback,
        )

    elif triggered_id == "load-button":
        try:
            g, parsed_log = ClusterState.from_string(load_graph_input, return_log=True)
            valid = True
            invalid = False
        except Exception as e:
            err_msg = str(e)
            return (
                no_update,
                no_update,
                no_update,
                f"Failed to load: {err_msg}",
                no_update,
                False,
                True,
                err_msg,
            )

        state = FrontendClusterState(num_nodes=len(g), cluster_state=g)
        move_log = parsed_log
        history_data = {"undo_stack": [], "redo_stack": []}
        cyto_data_new = state.to_cytoscape_elements()

        return (
            cyto_data_new,
            repr(g),
            move_log,
            "Action completed!",
            history_data,
            valid,
            invalid,
            feedback,
        )

    return (
        no_update,
        no_update,
        no_update,
        "Action completed!",
        history_data,
        valid,
        invalid,
        feedback,
    )


@callback(
    Output("download-json", "data"),
    Input("btn-get-json", "n_clicks"),
    State("figure-app", "elements"),
    prevent_initial_call=True,
)
def export_graph_to_json(n_clicks, cyto_data):
    if not n_clicks or not cyto_data:
        return no_update
    state = FrontendClusterState.from_cytoscape_elements(cyto_data)
    return dcc.send_string(state.to_json(), "cytoscape_graph.json")


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
