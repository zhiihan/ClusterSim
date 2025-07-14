from textwrap import dedent as d
from dash import dcc, html, Input, Output, State, callback, no_update
from cluster_sim.app import Grid
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
                    dbc.Button(
                        "Download Text",
                        id="download-stab-btn",
                        color="primary",
                        className="me-1",
                    ),
                    dcc.Download(
                        id="download-stab",
                    ),
                    dcc.Clipboard(
                        target_id="stabilizer-data",
                        style={
                            "fontSize": 20,
                        },
                    ),
                    html.Pre(
                        id="stabilizer-data",
                        style={
                            "border": "thin lightgrey solid",
                            "white-space": "pre",
                            "overflow": "auto",
                            "word-wrap": "normal",
                        },
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
    Output("download-stab", "data"),
    Input("download-stab-btn", "n_clicks"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    prevent_initial_call=True,
)
def download_stab(n_clicks, browserData, graphData):
    s = jsonpickle.decode(browserData)
    G = Grid(s.shape, json_data=graphData)
    G.graph.remove_nodes_from(list(nx.isolates(G.graph)))
    adjacency_matrix = nx.to_numpy_array(G.graph).astype(int)
    adjacency_matrix += np.diag(2 * np.ones(adjacency_matrix.shape[0])).astype(int)

    stabs = adjacency_mat_to_stabilizer(adjacency_matrix, newline=False)

    return dict(content=stabs, filename="stabilizers.txt")


@callback(
    Output("stabilizer-data", "children"),
    Output("browser-data", "data", allow_duplicate=True),
    Input("compute-stabilizers", "n_clicks"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    prevent_initial_call=True,
)
def stabilizer_data(n_clicks, browserData, graphData):
    """
    Updates stabilizer data.
    """

    s = jsonpickle.decode(browserData)
    G = Grid(s.shape, json_data=graphData)
    G.graph.remove_nodes_from(list(nx.isolates(G.graph)))

    adjacency_matrix = nx.to_numpy_array(G.graph).astype(int)

    adjacency_matrix += np.diag(2 * np.ones(adjacency_matrix.shape[0])).astype(int)

    s.stabilizer = adjacency_mat_to_stabilizer(adjacency_matrix)

    return html.P(s.stabilizer), s.to_json()


def adjacency_mat_to_stabilizer(adjacency_matrix, newline=True):
    """
    Convert the adjacency matrix to a stabilizer string.
    """
    program = ""

    for index, rows in enumerate(adjacency_matrix):
        instruction = np.array2string(
            rows, formatter={"int": array2pauli}, separator=""
        )
        instruction = instruction.replace("[", "").replace("]", "").replace(" ", "")

        program += instruction

        program += ","
        if newline:
            program += "\n"
    return program


def array2pauli(x):
    if x == 0:
        return "_"
    elif x == 1:
        return "Z"
    elif x == 2:
        return "X"
    elif x == 3:
        return "Y"
    else:
        raise ValueError("Invalid value in adjacency matrix")
