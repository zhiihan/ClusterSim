from cluster_sim import ClusterState
from textwrap import dedent as d
from dash import dcc, html, callback, Input, Output, State
import random
from cluster_sim.app import layouts, BrowserState
import dash_bootstrap_components as dbc
import logging

error_channel = dbc.Card(
    dbc.CardBody(
        [
            dcc.Markdown(
                d(
                    """
            **Apply Erasure Channel**

            Select a probability p to randomly remove nodes.
            """
                )
            ),
            dcc.Slider(
                0,
                0.3,
                step=0.03,
                value=0.06,
                tooltip={
                    "placement": "bottom",
                },
                marks={i / 100: str(i) for i in range(0, 31, 3)},
                id="p_err",
                className="dash-bootstrap",
            ),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button("Erasure Channel", id="reset-seed"),
                    ),
                    dbc.Col(
                        [
                            dbc.Input(
                                id="load-graph-seed",
                                type="number",
                                placeholder="Seed",
                            ),
                        ]
                    ),
                ]
            ),
        ]
    ),
)


@callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Input("reset-seed", "n_clicks"),
    State("load-graph-seed", "value"),
    State("p_err", "value"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    prevent_initial_call=True,
)
def apply_error_channel(nclicks, seed_input, p_err, browser_data, graphData):
    """
    Randomly measure qubits.
    """
    browser_state = BrowserState.from_json(browser_data)
    browser_state.p_err = p_err
    G = ClusterState.from_json(graphData)

    random.seed = seed_input
    logging.info(f"Loaded seed : {seed_input}, p = {browser_state.p_err}")
    ui = "Loaded seed : {}, p = {}".format(seed_input, browser_state.p_err)
    layout = layouts[browser_state.layout](
        browser_state=browser_state
    )
    num_qubits = len(G)

    for i in range(num_qubits):
        if random.random() < browser_state.p_err:
            if i not in browser_state.removed_nodes:
                browser_state.removed_nodes.add(i)
                # FIXME: Add VOP to log here
                G.measure(i, force=0, basis='Z')
                browser_state.log += f"{layout.get_node_coords(i)}, Z;\n"
                
    return browser_state.log, 1, ui, browser_state.to_json(), G.to_json()
