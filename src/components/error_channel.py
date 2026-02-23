from textwrap import dedent as d
from dash import dcc, html, callback, Input, Output, State
import random
from cluster_sim.app import BrowserState, layouts
from cluster_sim.simulator import ClusterState
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
                id="prob",
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
    State("prob", "value"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    prevent_initial_call=True,
)
def apply_error_channel(nclicks, seed_input, prob, browser_data, graphData):
    """
    Randomly measure qubits.
    """
    browser_state = BrowserState.from_json(browser_data)
    browser_state.p = prob

    G = ClusterState.from_json(graphData)
    if seed_input:
        # The user has inputted a seed
        random.seed(int(seed_input))
        logging.info(f"Loaded seed : {seed_input}, p = {browser_state.p}")
        ui = "Loaded seed : {}, p = {}".format(seed_input, browser_state.p)
    else:
        # Use a random seed.
        random.seed()
        logging.info(f"Loaded seed : {browser_state.seed}, p = {browser_state.p}")
        ui = "Loaded seed : None, p = {}, shape = {}".format(browser_state.p, browser_state.shape)
    # p is the probability of losing a qubit

    measurementChoice = "Z"
    layout =  layouts[browser_state.layout](graph = G.to_rustworkx(), browser_state=browser_state)

    for i in range(browser_state.xmax * browser_state.ymax * browser_state.zmax):
        if random.random() < browser_state.p:
            if i not in browser_state.removed_nodes:
                G.measure(i, measurementChoice)
                coords = layout.get_node_coords(i)
                browser_state.log += f"{coords}, {measurementChoice};\n"
    return (
        browser_state.log,
        1,
        ui,
        browser_state.to_json(),
        G.to_json(),
    )

