from textwrap import dedent as d
from dash import dcc, html, Input, Output, State, callback, no_update
from cluster_sim.app.grid import Grid
import dash_bootstrap_components as dbc
import jsonpickle
import networkx as nx
import json
import numpy as np

stabilizer = dbc.Card(
    dbc.CardBody(
        [
            dcc.Markdown(
                d(
                    """
**Stabilizer**

Click on points in the graph. Can be copied to clipboard to load a graph state.
"""
                )
            ),
            html.Div(
                [
                    dbc.Button(
                        "Compute Stabilizers",
                        id="compute-stabilizers",
                        color="primary",
                        className="me-1",
                    ),
                    dcc.Clipboard(
                        target_id="stabilizer-data",
                        style={
                            "fontSize": 20,
                        },
                    ),
                    html.Pre(
                        id="stabilizer-data",
                        style={"border": "thin lightgrey solid", "overflowX": "scroll"},
                    ),
                ],
                style={
                    "height": "400px",
                    "overflowY": "scroll",
                },
            ),
        ]
    )
)


@callback(
    Output("stabilizer-data", "children"),
    Input("compute-stabilizers", "n_clicks"),
    State("graph-data", "data"),
    State("browser-data", "data"),
    prevent_initial_call=True,
)
def stabilizer_data(n_clicks, graphData, browserData):
    """
    Updates stabilizer data.
    """

    s = jsonpickle.decode(browserData)
    G = Grid(s.shape, json_data=graphData)
    graph = G.graph.remove_nodes_from(list(nx.isolates(G.graph)))

    adjacency_matrix = nx.to_numpy_array(G.graph).astype(int)

    adjacency_matrix += np.diag(2 * np.ones(adjacency_matrix.shape[0])).astype(int)

    program = ""

    for index, rows in enumerate(adjacency_matrix):
        instruction = np.array2string(
            rows, formatter={"int": array2pauli}, separator=""
        )
        instruction = instruction.replace("[", "").replace("]", "").replace(" ", "")

        program += instruction
        program += "\n"

    print(program)

    s.stabilizer = program

    return html.P(s.stabilizer)


def array2pauli(x):
    if x == 0:
        return "I"
    elif x == 1:
        return "Z"
    elif x == 2:
        return "X"
    elif x == 3:
        return "Y"
    else:
        raise ValueError("Invalid value in adjacency matrix")
