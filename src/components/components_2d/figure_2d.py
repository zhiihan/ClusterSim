from dash import dcc, html, callback, Input, Output, State, no_update
from textwrap import dedent as d
import dash_bootstrap_components as dbc
from cluster_sim.simulator import ClusterState
import dash_cytoscape as cyto
from components.components_2d import (
    qubit_panel,
    postprocess_cyto_data_elements,
    preprocess_cyto_data_elements,
    move_log,
    fusion_menu,
)
from typing import List, Dict, Any
import random

import logging

# SVG Export
cyto.load_extra_layouts()


# Initialize the figure of the user's browsing section
def _init_state():
    G = ClusterState(5)

    for i in range(5):
        G.H(i)

    for i in range(5):
        G.CZ(i, (i + 1) % 5)

    cyto_data = G.to_cytoscape(export_elements=True)
    positions = [
        {"x": random.randint(0, 300), "y": random.randint(0, 300)}
        for i in range(len(G))
    ]
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


tab_1 = dbc.Row(
    [
        dbc.Col(qubit_panel, width=12)
    ]
)

tab_2 = dbc.Row(
    [
        dbc.Col(fusion_menu, width=12)
    ]
)

tab_3 = dbc.Row(
    [
        # Column 1: Load Input and Action Buttons
        dbc.Col(
            [
                dcc.Markdown("**Load Graph State**", style={"margin": "0 0 4px 0"}),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Input(
                                type="text",
                                id="load-graph-input",
                                placeholder="Load a graph state",
                                value="",
                                className="mb-0"
                            ),
                            width=6
                        ),
                        dbc.Col(dbc.Button("Load", id="load-button", style={"width": "100%"}), width=2),
                        dbc.Col(dbc.Button("Undo", id="undo-button", style={"width": "100%"}), width=2),
                        dbc.Col(dbc.Button("Redo", id="redo-button", style={"width": "100%"}), width=2),
                    ],
                    className="align-items-center"
                )
            ],
            width=12
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
            width=4
        ),
        dbc.Col(
            [
                dcc.Markdown("**Select Node Labels**", style={"margin": "0 0 4px 0"}),
                node_labels,
            ],
            width=4
        ),
        dbc.Col(
            [
                dcc.Markdown("**Grid Actions**", style={"margin": "0 0 4px 0"}),
                dbc.Button("Snap to Grid", id="snap-to-grid", style={"width": "100%"}),
            ],
            width=4,
            className="d-flex flex-column justify-content-end"
        )
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
                                            html.Button("×", id="close-modal-btn", className="btn-close", style={"fontSize": "24px", "background": "none", "border": "none", "color": "#64748b"}),
                                        ],
                                        style={"marginBottom": "20px"}
                                    ),
                                    dbc.Nav(
                                        [
                                            dbc.NavLink("Simulator Representation", id="modal-tab-rep", active=True, style={"cursor": "pointer"}),
                                            dbc.NavLink("Move Log", id="modal-tab-log", style={"cursor": "pointer"}),
                                            dbc.NavLink("Clifford Reference", id="modal-tab-ref", style={"cursor": "pointer"}),
                                        ],
                                        vertical=True,
                                        pills=True,
                                        className="flex-column"
                                    ),
                                ],
                                width=3,
                                style={"borderRight": "1px solid #dee2e6", "minHeight": "420px", "paddingRight": "20px"}
                            ),
                            # Modal Content Column
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            # Page 1: Simulator Representation
                                            html.Div(
                                                [
                                                    dcc.Markdown("**Simulator Representation**", style={"marginBottom": "8px"}),
                                                    html.Pre(
                                                        id="simulator-representation",
                                                        style={"whiteSpace": "pre-wrap", "height": "380px", "overflowY": "auto", "border": "1px solid #dee2e6", "borderRadius": "4px", "padding": "12px", "backgroundColor": "#f8f9fa", "color": "#212529", "fontFamily": "SFMono-Regular, Menlo, Monaco, Consolas, monospace", "fontSize": "13px"},
                                                        children=_init_state()[1],
                                                    ),
                                                ],
                                                id="modal-content-rep",
                                                style={"display": "block"}
                                            ),
                                            # Page 2: Move Log
                                            html.Div(
                                                move_log,
                                                id="modal-content-log",
                                                style={"display": "none"}
                                            ),
                                            # Page 3: Clifford Reference
                                            html.Div(
                                                [
                                                    dcc.Markdown("**Local Clifford Gate Decomposition Reference**", style={"marginBottom": "8px"}),
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
                                                        style={"height": "380px", "overflowY": "auto", "border": "1px solid #dee2e6", "borderRadius": "4px", "padding": "10px", "backgroundColor": "#f8f9fa"}
                                                    )
                                                ],
                                                id="modal-content-ref",
                                                style={"display": "none"}
                                            ),
                                        ],
                                        style={"paddingLeft": "20px"}
                                    )
                                ],
                                width=9
                            )
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
                dbc.Col(
                    html.H2("Cluster Sim", style={"margin": "0", "fontSize": "22px", "fontWeight": "bold"}),
                    width=2,
                    className="d-flex align-items-center"
                ),
                dbc.Col(
                    dcc.Loading(
                        dbc.Alert(
                            "Click on the graph to measure nodes.",
                            color="primary",
                            id="ui",
                            style={"padding": "8px 16px", "margin": "0"}
                        ),
                        delay_show=100,
                        delay_hide=100,
                        custom_spinner=dbc.Spinner(color="primary", size="sm"),
                    ),
                    width=4,
                    className="d-flex align-items-center"
                ),
                dbc.Col(
                    dbc.Tabs(
                        [
                            dbc.Tab(label="Operations", tab_id="tab-1"),
                            dbc.Tab(label="Fusion", tab_id="tab-2"),
                            dbc.Tab(label="Reset and Load", tab_id="tab-3"),
                            dbc.Tab(label="Plot Options", tab_id="tab-5"),
                        ],
                        id="tabs",
                        active_tab="tab-1",
                        style={"border": "none", "margin": "0", "justifyContent": "flex-end"}
                    ),
                    width=4,
                    className="d-flex align-items-center justify-content-end"
                ),
                dbc.Col(
                    dbc.Button("Reference & Logs", id="open-modal-btn", outline=True, color="secondary", style={"width": "100%"}),
                    width=2,
                    className="d-flex align-items-center justify-content-end"
                )
            ],
            className="align-items-center mb-2"
        ),
        html.Hr(style={"margin": "8px 0"}),
        html.Div(tab_1, id="tab-content-1", style={"display": "block"}),
        html.Div(tab_2, id="tab-content-2", style={"display": "none"}),
        html.Div(tab_3, id="tab-content-3", style={"display": "none"}),
        html.Div(tab_5, id="tab-content-5", style={"display": "none"}),
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

    label_style = [
        {
            "selector": "node",
            "style": {
                "label": node_labels,
                "color": "#212529",
                "font-size": "13px",
                "font-family": "system-ui, -apple-system, sans-serif",
                "font-weight": "bold",
                "text-valign": "center",
                "text-halign": "center",
                "width": "38px",
                "height": "38px",
                "border-width": "2px",
                "border-color": "#ffffff",
            }
        },
        {
            "selector": "edge",
            "style": {
                "width": 3,
                "line-color": "rgba(0, 0, 0, 0.22)",
                "curve-style": "bezier",
            }
        }
    ]

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
                },
            }
        )

    selected_style = [
        {
            "selector": "node:selected",
            "style": {
                "border-width": "4px",
                "border-color": "#0d6efd",
                "width": "42px",
                "height": "42px",
            },
        },
        {
            "selector": "edge:selected",
            "style": {
                "line-color": "#0d6efd",
                "width": 5,
            }
        }
    ]

    return label_style + color_styles + selected_style


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
    Output("tab-content-1", "style"),
    Output("tab-content-2", "style"),
    Output("tab-content-3", "style"),
    Output("tab-content-5", "style"),
    Input("tabs", "active_tab"),
)
def toggle_tab_visibility(active_tab):
    styles = [{"display": "none"} for _ in range(4)]
    if active_tab == "tab-1":
        styles[0] = {"display": "block"}
    elif active_tab == "tab-2":
        styles[1] = {"display": "block"}
    elif active_tab == "tab-3":
        styles[2] = {"display": "block"}
    elif active_tab == "tab-5":
        styles[3] = {"display": "block"}
    return styles[0], styles[1], styles[2], styles[3]


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
    Input("modal-tab-rep", "n_clicks"),
    Input("modal-tab-log", "n_clicks"),
    Input("modal-tab-ref", "n_clicks"),
    prevent_initial_call=True,
)
def switch_modal_tab(n_rep, n_log, n_ref):
    ctx = callback_context
    if not ctx.triggered:
        return "rep", True, False, False
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "modal-tab-log":
        return "log", False, True, False
    elif trigger_id == "modal-tab-ref":
        return "ref", False, False, True
    return "rep", True, False, False


@callback(
    Output("modal-content-rep", "style"),
    Output("modal-content-log", "style"),
    Output("modal-content-ref", "style"),
    Input("modal-tab-store", "data"),
)
def toggle_modal_content_visibility(active_tab):
    styles = [{"display": "none"} for _ in range(3)]
    if active_tab == "rep":
        styles[0] = {"display": "block"}
    elif active_tab == "log":
        styles[1] = {"display": "block"}
    elif active_tab == "ref":
        styles[2] = {"display": "block"}
    return styles[0], styles[1], styles[2]
