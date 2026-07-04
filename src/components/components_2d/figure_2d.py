from dash import dcc, html, callback, Input, Output, State, no_update, callback_context
from textwrap import dedent as d
import dash_bootstrap_components as dbc
from clustersim.simulator import ClusterState
import dash_cytoscape as cyto
from components.components_2d import (
    qubit_panel,
    postprocess_cyto_data_elements,
    preprocess_cyto_data_elements,
    move_log,
)
from typing import List, Dict, Any
import random

import logging

# SVG Export
cyto.load_extra_layouts()


# Initialize the figure of the user's browsing section
def _init_state():
    G = ClusterState(0)
    cyto_data = G.to_cytoscape(export_elements=True)
    positions = []
    cyto_data = postprocess_cyto_data_elements(cyto_data, positions)
    return cyto_data, repr(G)


figure_2d = cyto.Cytoscape(
    id="figure-app",
    layout={"name": "preset"},
    style={"width": "100%", "height": "100%"},
    stylesheet=[{"selector": "node", "style": {"label": "data(vop)"}}],
    elements=_init_state()[0],
    selectedNodeData=[],
    boxSelectionEnabled=True,
    wheelSensitivity=0.2,
    minZoom=0.2,
    maxZoom=20,
)

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
    placeholder="Value",
    value="data(value)",
)


tab_1 = dbc.Row([dbc.Col(qubit_panel, width=12)])

tab_3 = dbc.Row(
    [
        # Column 1: Load Input and Action Buttons
        dbc.Col(
            [
                dcc.Markdown("**Load Graph State**", style={"margin": "0 0 4px 0"}),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Input(
                                    type="text",
                                    id="load-graph-input",
                                    placeholder="Load a graph state",
                                    value="",
                                    className="mb-0",
                                ),
                                dbc.FormFeedback(
                                    "Graph loaded successfully!",
                                    type="valid",
                                    id="load-feedback-valid",
                                ),
                                dbc.FormFeedback(
                                    "", type="invalid", id="load-feedback-invalid"
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(dbc.Button("Load", id="load-button", style={"width": "100%"}), width=2),
                        dbc.Col(dbc.Button("Undo", id="undo-button", style={"width": "100%"}), width=2),
                        dbc.Col(dbc.Button("Redo", id="redo-button", style={"width": "100%"}), width=2),
                        dbc.Tooltip("Undo (Ctrl+Z)", target="undo-button", placement="top"),
        dbc.Tooltip("Redo (Ctrl+Y)", target="redo-button", placement="top"),
                    ],
                    className="align-items-center",
                ),
            ],
            width=12,
        )
    ]
)

tab_5 = dbc.Row(
    [
        dbc.Col(
            [
                dcc.Markdown("**Select Layout**", style={"margin": "0 0 4px 0"}),
                layout_dropdown,
            ],
            width=4,
        ),
        dbc.Col(
            [
                dcc.Markdown("**Select Node Labels**", style={"margin": "0 0 4px 0"}),
                node_labels,
            ],
            width=4,
        ),
        dbc.Col(
            [
                dcc.Markdown("**Plot Options**", style={"margin": "0 0 4px 0"}),
                dbc.Button("Customize Plot", id="open-offcanvas-btn-tab5", color="secondary", outline=True, style={"width": "100%", "marginBottom": "4px"}),
                dbc.Button("Snap to Grid", id="snap-to-grid", style={"width": "100%"}),
                dbc.Button("Export SVG", id="btn-get-svg"),
                dbc.Button("Export JSON", id="btn-get-json"),
                dcc.Download(id="download-json"),
            ],
            width=4,
            className="d-flex flex-column justify-content-end",
        ),
    ]
)

tab_ui_2d = html.Div(
    [
        # Store for active modal tab
        dcc.Store(id="modal-tab-store", data="rep"),
        # Reference & Logs Modal
        dbc.Modal(
            [
                dbc.ModalBody(
                    dbc.Row(
                        [
                            # Modal Sidebar Navigation Column
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.Button(
                                                "×",
                                                id="close-modal-btn",
                                                className="btn-close",
                                                style={
                                                    "fontSize": "24px",
                                                    "background": "none",
                                                    "border": "none",
                                                    "color": "#64748b",
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "20px"},
                                    ),
                                    dbc.Nav(
                                        [
                                            dbc.NavLink(
                                                "Simulator Representation",
                                                id="modal-tab-rep",
                                                active=True,
                                                style={"cursor": "pointer"},
                                            ),
                                            dbc.NavLink(
                                                "Move Log",
                                                id="modal-tab-log",
                                                style={"cursor": "pointer"},
                                            ),
                                            dbc.NavLink(
                                                "Clifford Reference",
                                                id="modal-tab-ref",
                                                style={"cursor": "pointer"},
                                            ),
                                            html.Hr(style={"margin": "8px 0"}),
                                            dbc.NavLink(
                                                "Reset & Load",
                                                id="modal-tab-reset",
                                                style={"cursor": "pointer"},
                                            ),
                                            dbc.NavLink(
                                                "Plot Options",
                                                id="modal-tab-plot",
                                                style={"cursor": "pointer"},
                                            ),
                                        ],
                                        vertical=True,
                                        pills=True,
                                        className="flex-column",
                                    ),
                                ],
                                width=3,
                                style={
                                    "borderRight": "1px solid #dee2e6",
                                    "minHeight": "420px",
                                    "paddingRight": "20px",
                                },
                            ),
                            # Modal Content Column
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            # Page 1: Simulator Representation
                                            html.Div(
                                                [
                                                    dcc.Markdown(
                                                        "**Simulator Representation**",
                                                        style={"marginBottom": "8px"},
                                                    ),
                                                    html.Pre(
                                                        id="simulator-representation",
                                                        style={
                                                            "whiteSpace": "pre-wrap",
                                                            "height": "380px",
                                                            "overflowY": "auto",
                                                            "border": "1px solid #dee2e6",
                                                            "borderRadius": "4px",
                                                            "padding": "12px",
                                                            "backgroundColor": "#f8f9fa",
                                                            "color": "#212529",
                                                            "fontFamily": "SFMono-Regular, Menlo, Monaco, Consolas, monospace",
                                                            "fontSize": "13px",
                                                        },
                                                        children=_init_state()[1],
                                                    ),
                                                ],
                                                id="modal-content-rep",
                                                style={"display": "block"},
                                            ),
                                            # Page 2: Move Log
                                            html.Div(
                                                move_log,
                                                id="modal-content-log",
                                                style={"display": "none"},
                                            ),
                                            # Page 3: Clifford Reference
                                            html.Div(
                                                [
                                                    dcc.Markdown(
                                                        "**Local Clifford Gate Decomposition Reference**",
                                                        style={"marginBottom": "8px"},
                                                    ),
                                                    html.Div(
                                                        dcc.Markdown(
                                                            d("""
                                                            ```python
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
                                                        ),
                                                        style={
                                                            "height": "380px",
                                                            "overflowY": "auto",
                                                            "border": "1px solid #dee2e6",
                                                            "borderRadius": "4px",
                                                            "padding": "10px",
                                                            "backgroundColor": "#f8f9fa",
                                                        },
                                                    ),
                                                ],
                                                id="modal-content-ref",
                                                style={"display": "none"},
                                            ),
                                            # Page 4: Reset & Load
                                            html.Div(
                                                tab_3,
                                                id="modal-content-reset",
                                                style={"display": "none"},
                                            ),
                                            # Page 5: Plot Options
                                            html.Div(
                                                tab_5,
                                                id="modal-content-plot",
                                                style={"display": "none"},
                                            ),
                                        ],
                                        style={"paddingLeft": "20px"},
                                    )
                                ],
                                width=9,
                            ),
                        ]
                    )
                )
            ],
            id="reference-modal",
            size="xl",
            is_open=False,
            centered=True,
        ),
        # Main Header Row
        dbc.Row(
            [
                # Left: Title + Alert stacked
                dbc.Col(
                    [
                        html.H2(
                            "Cluster Sim",
                            style={
                                "margin": "0 0 4px 0",
                                "fontSize": "22px",
                                "fontWeight": "bold",
                            },
                        ),
                        dcc.Loading(
                            dbc.Alert(
                                "Click on the graph to measure nodes.",
                                color="primary",
                                id="ui",
                                style={
                                    "padding": "4px 10px",
                                    "margin": "0",
                                    "fontSize": "0.8rem",
                                },
                            ),
                            delay_show=100,
                            delay_hide=100,
                            custom_spinner=dbc.Spinner(color="primary", size="sm"),
                        ),
                    ],
                    width=2,
                    className="d-flex flex-column justify-content-center",
                ),
                # Center: Operations panel
                dbc.Col(
                    tab_1,
                    width=10,
                ),
            ],
            className="align-items-start mb-2 flex-nowrap",
        ),
    ]
)


@callback(
    Output("ui", "children", allow_duplicate=True),
    Input("figure-app", "selectedNodeData"),
    prevent_initial_call=True,
)
def displaySelectedNodeData(data_list: List[Dict[str, Any]]):
    if not data_list:
        return "Click on the graph to select nodes, or SHIFT+click to select multiple nodes."
    else:
        logging.debug(f"Selected {[i['value'] for i in data_list]}")
        return f"Selected {[i['value'] for i in data_list]}"


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
    Input("node_labels", "value"),
)
def update_stylesheet(_, node_labels: str):

    label_style = [{"selector": "node", "style": {"label": node_labels}}]

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
        color_styles.append(
            {
                "selector": f'[vop *= "{label}"]',
                "style": {
                    "background-color": color,
                    "border-width": 2,
                    "border-color": "#000000",
                },
            }
        )

    selected_style = [
        {
            "selector": "node:selected",
            "style": {
                "border-width": 3,
                "border-color": "blue",
            },
        }
    ]

    custom_styles = [
        {
            "selector": "node",
            "style": {
                "shape": "data(shape)",
            },
        },
        {
            "selector": "edge",
            "style": {
                "label": "data(label)",
                "text-rotation": "autorotate",
                "text-margin-y": -10,
                "font-size": "10px",
                "color": "#475569",
            },
        },
        {
            "selector": "node:parent",
            "style": {
                "background-opacity": 0.333,
                "background-color": "#cbd5e1",
                "border-color": "#94a3b8",
                "border-width": 2,
                "label": "data(label)",
                "font-size": "14px",
                "font-weight": "bold",
                "text-valign": "top",
                "text-halign": "center",
                "text-margin-y": -6,
            },
        },
        {
            "selector": "node:childless[!vop]",
            "style": {
                "display": "none",
            },
        },
    ]

    return label_style + color_styles + selected_style + custom_styles


@callback(
    Output("figure-app", "elements", allow_duplicate=True),
    Input("snap-to-grid", "n_clicks"),
    State("figure-app", "elements"),
    prevent_initial_call=True,
)
def handle_snap_to_grid(n_clicks, cyto_data):
    if not n_clicks or not cyto_data:
        return no_update

    cyto_data_pre, positions = preprocess_cyto_data_elements(cyto_data)
    snapped_positions = [
        {"x": round(pos["x"] / 50) * 50, "y": round(pos["y"] / 50) * 50}
        for pos in positions
    ]
    return postprocess_cyto_data_elements(cyto_data, snapped_positions)


@callback(
    Output("reference-modal", "is_open"),
    Input("open-modal-btn", "n_clicks"),
    Input("close-modal-btn", "n_clicks"),
    State("reference-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_modal(n1, n2, is_open):
    return not is_open


@callback(
    Output("modal-tab-store", "data"),
    Output("modal-tab-rep", "active"),
    Output("modal-tab-log", "active"),
    Output("modal-tab-ref", "active"),
    Output("modal-tab-reset", "active"),
    Output("modal-tab-plot", "active"),
    Input("modal-tab-rep", "n_clicks"),
    Input("modal-tab-log", "n_clicks"),
    Input("modal-tab-ref", "n_clicks"),
    Input("modal-tab-reset", "n_clicks"),
    Input("modal-tab-plot", "n_clicks"),
    prevent_initial_call=True,
)
def switch_modal_tab(n_rep, n_log, n_ref, n_reset, n_plot):
    ctx = callback_context
    if not ctx.triggered:
        return "rep", True, False, False, False, False

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "modal-tab-log":
        return "log", False, True, False, False, False
    elif trigger_id == "modal-tab-ref":
        return "ref", False, False, True, False, False
    elif trigger_id == "modal-tab-reset":
        return "reset", False, False, False, True, False
    elif trigger_id == "modal-tab-plot":
        return "plot", False, False, False, False, True
    return "rep", True, False, False, False, False


@callback(
    Output("modal-content-rep", "style"),
    Output("modal-content-log", "style"),
    Output("modal-content-ref", "style"),
    Output("modal-content-reset", "style"),
    Output("modal-content-plot", "style"),
    Input("modal-tab-store", "data"),
)
def toggle_modal_content_visibility(active_tab):
    styles = [{"display": "none"} for _ in range(5)]
    if active_tab == "rep":
        styles[0] = {"display": "block"}
    elif active_tab == "log":
        styles[1] = {"display": "block"}
    elif active_tab == "ref":
        styles[2] = {"display": "block"}
    elif active_tab == "reset":
        styles[3] = {"display": "block"}
    elif active_tab == "plot":
        styles[4] = {"display": "block"}
    return styles[0], styles[1], styles[2], styles[3], styles[4]
