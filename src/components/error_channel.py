from textwrap import dedent as d
from dash import dcc, html, callback, Input, Output, State
import jsonpickle
import random
from cluster_sim.app import get_node_coords
from cluster_sim.simulator import ClusterState, NetworkXState
import dash_bootstrap_components as dbc
import logging
import networkx as nx

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
    Output("holes-data", "data", allow_duplicate=True),
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
    s = jsonpickle.decode(browser_data)
    s.p = prob

    G = ClusterState.from_json(graphData)
    D = NetworkXState(nx.Graph())
    if seed_input:
        # The user has inputted a seed
        random.seed(int(seed_input))
        logging.info(f"Loaded seed : {seed_input}, p = {s.p}")
        ui = "Loaded seed : {}, p = {}".format(seed_input, s.p)
    else:
        # Use a random seed.
        random.seed()
        logging.info(f"Loaded seed : {s.seed}, p = {s.p}")
        ui = "Loaded seed : None, p = {}, shape = {}".format(s.p, s.shape)
    # p is the probability of losing a qubit

    measurementChoice = "Z"

    for i in range(s.xmax * s.ymax * s.zmax):
        if random.random() < s.p:
            if not s.removed_nodes[i]:
                s.removed_nodes[i] = True
                G.measure(i, measurementChoice)
                s.log.append(f"{get_node_coords(i, s.shape)}, {measurementChoice}; ")
                s.log.append(html.Br())
                s.move_list.append([get_node_coords(i, s.shape), measurementChoice])
                D.add_node(i)
    return s.log, 1, ui, s.to_json(), G.to_json(), D.to_json()
