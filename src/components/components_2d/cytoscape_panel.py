import uuid
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State, no_update, callback_context
from clustersim.simulator import FrontendClusterState

# Preset shapes supported by Cytoscape JS
SHAPE_OPTIONS = [
    {"label": "Ellipse (Default)", "value": "ellipse"},
    {"label": "Rectangle", "value": "rectangle"},
    {"label": "Round Rectangle", "value": "round-rectangle"},
    {"label": "Triangle", "value": "triangle"},
    {"label": "Pentagon", "value": "pentagon"},
    {"label": "Hexagon", "value": "hexagon"},
    {"label": "Heptagon", "value": "heptagon"},
    {"label": "Octagon", "value": "octagon"},
    {"label": "Star", "value": "star"},
    {"label": "Diamond", "value": "diamond"},
    {"label": "Vee", "value": "vee"},
    {"label": "Rhombus", "value": "rhombus"},
]

layout_dropdown = dcc.Dropdown(
    ["random", "grid", "circle", "breadthfirst", "cose", "concentric", "preset"],
    "preset",
    id="graph_layout_dropdown",
)

node_labels = dbc.Select(
    id="node_labels",
    options=[
        {"label": "Value", "value": "data(value)"},
        {"label": "ID", "value": "data(id)"},
        {"label": "Label", "value": "data(label)"},
        {"label": "VOP", "value": "data(vop)"},
    ],
    placeholder="Label",
    value="data(label)",
    size="sm",
)

# Offcanvas UI component definition
cytoscape_offcanvas = dbc.Offcanvas(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Graph Layout", className="form-label small fw-bold mb-1"
                        ),
                        layout_dropdown,
                    ],
                    className="mb-3",
                ),
                html.Div(
                    [
                        html.Label(
                            "Node Labels Display",
                            className="form-label small fw-bold mb-1",
                        ),
                        node_labels,
                    ],
                    className="mb-3",
                ),
                html.Hr(className="my-3"),
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Custom Label", className="form-label small fw-bold mb-1"
                        ),
                        dbc.Input(
                            id="custom-label-input",
                            type="text",
                            placeholder="Enter label text...",
                            size="sm",
                        ),
                    ],
                    className="mb-2",
                ),
                html.Div(
                    [
                        html.Label(
                            "Node Shape", className="form-label small fw-bold mb-1"
                        ),
                        dbc.Select(
                            id="node-shape-select",
                            options=SHAPE_OPTIONS,
                            value="ellipse",
                            size="sm",
                            placeholder="Ellipse (Default)",
                        ),
                    ],
                    className="mb-3",
                ),
                html.Div(
                    [
                        dbc.Button(
                            "Apply Label",
                            id="apply-label-btn",
                            color="primary",
                            size="sm",
                            className="w-100 mb-2",
                        ),
                        dbc.Button(
                            "Apply Shape",
                            id="apply-shape-btn",
                            color="primary",
                            size="sm",
                            className="w-100 mb-2",
                        ),
                        dbc.Button(
                            "Create Group from Selection",
                            id="create-group-btn",
                            color="primary",
                            size="sm",
                            className="w-100 mb-2",
                        ),
                    ],
                    className="d-flex flex-column mb-3",
                ),
                html.Hr(className="my-3"),
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        dbc.Button(
                            "Snap to Grid",
                            id="snap-to-grid",
                            color="secondary",
                            outline=True,
                            size="sm",
                            className="w-100 mb-2",
                        ),
                        dbc.Button(
                            "Export SVG",
                            id="btn-get-svg",
                            color="secondary",
                            outline=True,
                            size="sm",
                            className="w-100 mb-2",
                        ),
                        dbc.Button(
                            "Export JSON",
                            id="btn-get-json",
                            color="secondary",
                            outline=True,
                            size="sm",
                            className="w-100 mb-2",
                        ),
                        dcc.Download(id="download-json"),
                    ],
                    className="d-flex flex-column",
                ),
            ]
        ),
    ],
    id="cytoscape-offcanvas",
    title="Plot Options & Customization",
    is_open=False,
    scrollable=True,
    backdrop=False,
    placement="end",
    style={"width": "400px"},
)


@callback(
    Output("cytoscape-offcanvas", "is_open"),
    Input("open-offcanvas-btn", "n_clicks"),
    State("cytoscape-offcanvas", "is_open"),
    prevent_initial_call=True,
)
def toggle_offcanvas(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


@callback(
    Output("custom-label-input", "value"),
    Output("node-shape-select", "value"),
    Output("create-group-btn", "children"),
    Input("figure-app", "selectedNodeData"),
    Input("figure-app", "selectedEdgeData"),
    State("figure-app", "elements"),
)
def update_offcanvas_inputs(selected_nodes, selected_edges, elements):
    label_val = ""
    shape_val = no_update
    btn_text = "Create Group from Selection"

    if selected_nodes and elements:
        state = FrontendClusterState.from_cytoscape_elements(elements)
        is_grouped = False
        for node in selected_nodes:
            n_id = str(node.get("id"))
            if n_id in state.groups:
                is_grouped = True
                break
            q = state.resolve_qubit(n_id)
            if q is not None and state.get_group(q) is not None:
                is_grouped = True
                break
        if is_grouped:
            btn_text = "Ungroup Selection"

    if selected_nodes and len(selected_nodes) == 1:
        node = selected_nodes[0]
        node_id = str(node.get("id"))
        actual = next(
            (
                el
                for el in (elements or [])
                if str(el.get("data", {}).get("id")) == node_id
            ),
            None,
        )
        data = actual.get("data", {}) if actual else node

        label_val = data.get("label", "")
        if "vop" in data and label_val == str(data.get("value", data.get("id"))):
            label_val = ""
        shape_val = data.get("shape") or (
            actual.get("style", {}).get("shape") if actual else "ellipse"
        )

    elif selected_edges and len(selected_edges) == 1:
        edge = selected_edges[0]
        src, tgt = str(edge.get("source")), str(edge.get("target"))
        actual = next(
            (
                el
                for el in (elements or [])
                if str(el.get("data", {}).get("source")) == src
                and str(el.get("data", {}).get("target")) == tgt
            ),
            None,
        )
        data = actual.get("data", {}) if actual else edge
        label_val = data.get("label", "")

    return label_val, shape_val, btn_text


@callback(
    Output("figure-app", "elements", allow_duplicate=True),
    Output("history-store", "data", allow_duplicate=True),
    Input("apply-label-btn", "n_clicks"),
    Input("apply-shape-btn", "n_clicks"),
    Input("create-group-btn", "n_clicks"),
    State("figure-app", "elements"),
    State("figure-app", "selectedNodeData"),
    State("figure-app", "selectedEdgeData"),
    State("custom-label-input", "value"),
    State("node-shape-select", "value"),
    State("history-store", "data"),
    State("move-log", "children"),
    prevent_initial_call=True,
)
def handle_cytoscape_options(
    btn_label,
    btn_shape,
    btn_create_group,
    elements,
    selected_nodes,
    selected_edges,
    custom_label,
    node_shape,
    history_data,
    move_log,
):
    ctx = callback_context
    if not ctx.triggered or not elements:
        return no_update, no_update

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    lbl_text = custom_label.strip() if custom_label else ""

    if not history_data or not isinstance(history_data, dict):
        history_data = {"undo_stack": [], "redo_stack": []}
    else:
        history_data = {
            "undo_stack": list(history_data.get("undo_stack", [])),
            "redo_stack": list(history_data.get("redo_stack", [])),
        }

    state = FrontendClusterState.from_cytoscape_elements(elements)

    if trigger_id == "apply-label-btn":
        if selected_nodes:
            for n in selected_nodes:
                n_id = str(n.get("id"))
                if n_id in state.groups:
                    state.groups[n_id]["label"] = lbl_text if lbl_text else ""
                else:
                    q = state.resolve_qubit(n_id)
                    if q is not None and 0 <= q < len(state):
                        state.node_metadata[q]["label"] = (
                            lbl_text if lbl_text else str(q)
                        )

        # Record state before applying offcanvas customization
        history_data["undo_stack"].append(
            {
                "action_type": "frontend_customization",
                "move_log": move_log or "",
                "elements": elements,
            }
        )
        history_data["redo_stack"] = []
        return state.to_cytoscape_elements(), history_data

    elif trigger_id == "apply-shape-btn":
        # Record state before applying offcanvas customization
        history_data["undo_stack"].append(
            {
                "action_type": "frontend_customization",
                "move_log": move_log or "",
                "elements": elements,
            }
        )
        history_data["redo_stack"] = []
        if selected_nodes:
            for n in selected_nodes:
                n_id = str(n.get("id"))
                q = state.resolve_qubit(n_id)
                if q is not None and 0 <= q < len(state):
                    state.node_metadata[q]["shape"] = node_shape
                    if "style" not in state.node_metadata[q]:
                        state.node_metadata[q]["style"] = {}
                    state.node_metadata[q]["style"]["shape"] = node_shape
        return state.to_cytoscape_elements(), history_data

    elif trigger_id == "create-group-btn":
        if not selected_nodes:
            return no_update, no_update

        selected_group_ids = []
        selected_qubits = []

        for n in selected_nodes:
            n_id = str(n.get("id"))
            if n_id in state.groups:
                selected_group_ids.append(n_id)
            else:
                q = state.resolve_qubit(n_id)
                if q is not None:
                    selected_qubits.append(q)

        has_group_container = len(selected_group_ids) > 0
        grouped_qubits = [q for q in selected_qubits if state.get_group(q) is not None]
        is_ungroup_action = has_group_container or len(grouped_qubits) > 0

        # Record state before applying offcanvas customization
        history_data["undo_stack"].append(
            {
                "action_type": "frontend_customization",
                "move_log": move_log or "",
                "elements": elements,
            }
        )
        history_data["redo_stack"] = []

        if is_ungroup_action:
            # Ungroup action: remove selected group containers and unassign grouped qubits
            for g_id in selected_group_ids:
                state.remove_group(g_id, unassign_children=True)

            affected_groups = set()
            for q in grouped_qubits:
                g_id = state.get_group(q)
                if g_id:
                    affected_groups.add(g_id)
                    state.set_group(q, None)

            # Clean up any affected groups that no longer have member nodes
            for g_id in affected_groups:
                remaining = any(
                    meta.get("parent") == g_id for meta in state.node_metadata
                )
                if not remaining:
                    state.remove_group(g_id)
        else:
            # Group action: create a new group for selected un-grouped qubits
            if not selected_qubits:
                return no_update, no_update

            g_id = f"group_{uuid.uuid4().hex[:8]}"
            g_label = lbl_text if lbl_text else ""
            state.add_group(g_id, label=g_label)
            for q in selected_qubits:
                if q < len(state):
                    state.set_group(q, g_id)

        return state.to_cytoscape_elements(), history_data

    return no_update, no_update
