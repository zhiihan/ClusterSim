from textwrap import dedent as d
from dash import dcc, html

import dash_bootstrap_components as dbc

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
