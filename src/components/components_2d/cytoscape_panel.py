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

# Offcanvas UI component definition
cytoscape_offcanvas = dbc.Offcanvas(
    [
        html.Div(
            [
                html.Label("Label", className="form-label small fw-bold mb-1"),
                dbc.Input(id="custom-label-input", type="text", placeholder="Enter label text...", size="sm"),
            ],
            className="mb-3",
        ),
        html.Div(
            [
                html.Label("Node Shape", className="form-label small fw-bold mb-1"),
                dbc.Select(id="node-shape-select", options=SHAPE_OPTIONS, value="ellipse", size="sm"),
            ],
            className="mb-4",
        ),
        html.Div(
            [
                dbc.Button("Apply Label", id="apply-label-btn", color="primary", size="sm", className="w-100 mb-2"),
                dbc.Button("Apply Shape", id="apply-shape-btn", color="primary", size="sm", className="w-100 mb-2"),
                dbc.Button("Create Group from Selection", id="create-group-btn", color="primary", size="sm", className="w-100 mb-2"),
                dbc.Button("Delete Selected Group", id="remove-group-btn", color="primary", size="sm", className="w-100 mb-4"),
            ],
            className="d-flex flex-column",
        ),
    ],
    id="cytoscape-offcanvas",
    title="Plot Customization",
    is_open=False,
    scrollable=True,
    backdrop=False,
    placement="end",
    style={"width": "400px"},
)


@callback(
    Output("cytoscape-offcanvas", "is_open"),
    Input("open-offcanvas-btn", "n_clicks"),
    Input("open-offcanvas-btn-tab5", "n_clicks"),
    State("cytoscape-offcanvas", "is_open"),
    prevent_initial_call=True,
)
def toggle_offcanvas(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@callback(
    Output("custom-label-input", "value"),
    Output("node-shape-select", "value"),
    Input("figure-app", "selectedNodeData"),
    Input("figure-app", "selectedEdgeData"),
    State("figure-app", "elements"),
)
def update_offcanvas_inputs(selected_nodes, selected_edges, elements):
    label_val = ""
    shape_val = "ellipse"

    if selected_nodes and len(selected_nodes) == 1:
        node = selected_nodes[0]
        node_id = str(node.get("id"))
        actual = next((el for el in (elements or []) if str(el.get("data", {}).get("id")) == node_id), None)
        data = actual.get("data", {}) if actual else node

        label_val = data.get("label", "")
        if "vop" in data and label_val == str(data.get("value", data.get("id"))):
            label_val = ""
        shape_val = data.get("shape", "ellipse")

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

    return label_val, shape_val


@callback(
    Output("figure-app", "elements", allow_duplicate=True),
    Output("history-store", "data", allow_duplicate=True),
    Input("apply-label-btn", "n_clicks"),
    Input("apply-shape-btn", "n_clicks"),
    Input("create-group-btn", "n_clicks"),
    Input("remove-group-btn", "n_clicks"),
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
    btn_remove_group,
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

    if not history_data or not isinstance(history_data, dict):
        history_data = {"undo_stack": [], "redo_stack": []}
    else:
        history_data = {
            "undo_stack": list(history_data.get("undo_stack", [])),
            "redo_stack": list(history_data.get("redo_stack", [])),
        }

    # Record state before applying offcanvas customization
    history_data["undo_stack"].append({"move_log": move_log or "", "elements": elements})
    history_data["redo_stack"] = []

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    lbl_text = custom_label.strip() if custom_label else ""

    state = FrontendClusterState.from_cytoscape_elements(elements)

    if trigger_id == "apply-label-btn":
        if selected_nodes:
            selected_ids = {str(n["id"]) for n in selected_nodes}
            for i in range(len(state)):
                if str(i) in selected_ids:
                    state.node_metadata[i]["label"] = lbl_text if lbl_text else str(state.get_value(i))
            for g_id in state.groups:
                if g_id in selected_ids and lbl_text:
                    state.groups[g_id]["label"] = lbl_text
        elif selected_edges:
            # edge label update in elements
            pass

    elif trigger_id == "apply-shape-btn":
        if selected_nodes:
            selected_ids = {str(n["id"]) for n in selected_nodes}
            for i in range(len(state)):
                if str(i) in selected_ids:
                    if "style" not in state.node_metadata[i]:
                        state.node_metadata[i]["style"] = {}
                    state.node_metadata[i]["style"]["shape"] = node_shape

    elif trigger_id == "create-group-btn":
        if not selected_nodes:
            return no_update, no_update
        selected_qubits = []
        for n in selected_nodes:
            try:
                selected_qubits.append(int(n["id"]))
            except (ValueError, KeyError):
                pass

        if not selected_qubits:
            return no_update, no_update

        g_id = f"group_{uuid.uuid4().hex[:8]}"
        g_label = lbl_text if lbl_text else f"Group {g_id[-8:]}"
        state.add_group(g_id, label=g_label)
        for q in selected_qubits:
            if q < len(state):
                state.set_group(q, g_id)

    elif trigger_id == "remove-group-btn":
        if not selected_nodes:
            return no_update, no_update
        groups_to_remove = set()
        for n in selected_nodes:
            nid = str(n["id"])
            if nid in state.groups:
                groups_to_remove.add(nid)
            else:
                try:
                    q = int(nid)
                    grp = state.get_group(q)
                    if grp:
                        groups_to_remove.add(grp)
                except ValueError:
                    pass

        for g_id in groups_to_remove:
            state.remove_group(g_id, unassign_children=True)

    return state.to_cytoscape_elements(), history_data
