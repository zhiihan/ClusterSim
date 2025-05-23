from textwrap import dedent as d
from dash import dcc, html, callback, Input, Output, State
from cluster_sim.app.grid import Grid
from cluster_sim.app.holes import Holes
from cluster_sim.app.state import BrowserState
from cluster_sim.app.utils import (
    get_node_index,
    get_node_coords,
)
import re

import dash_bootstrap_components as dbc
import jsonpickle
from dash import no_update
import numpy as np

load_graph = dbc.Card(
    dbc.CardBody(
        [
            dcc.Markdown(
                d(
                    """
                **Load Graph State**

                Paste data to load a graph state.
                """
                )
            ),
            dbc.Input(
                id="load-graph-input",
                type="text",
                placeholder="Load Graph State",
            ),
            html.Hr(),
            dbc.Button("Load Graph", id="load-graph-button"),
            # dcc.Store stores the intermediate value
            dcc.Store(id="browser-data"),
            dcc.Store(id="graph-data"),
            dcc.Store(id="holes-data"),
            dcc.Store(id="draw-plot"),
            html.Div(
                id="none",
                children=[],
                style={"display": "none"},
            ),
        ]
    )
)


def process_string(input_string):
    regex = re.compile(
        r"\((?P<x>[0-9]+),\s?(?P<y>[0-9]+),\s?(?P<z>[0-9]+)\),\s?(?P<basis>[XYZ]);\s?"
    )

    regex_matches = regex.findall(input_string)

    instructions = []

    for match in regex_matches:
        print(f"Match: {match}")
        x = int(match[0])
        y = int(match[1])
        z = int(match[2])
        basis = match[3]

        instructions.append(((x, y, z), basis))

        if basis not in ["X", "Y", "Z"]:
            raise ValueError(f"Invalid basis: {basis}")

    return instructions


@callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("load-graph-button", "n_clicks"),
    State("load-graph-input", "value"),
    State("browser-data", "data"),
    prevent_initial_call=True,
)
def load_graph_from_string(n_clicks, input_string, browser_data):

    if input_string is None:
        return no_update, no_update, no_update, no_update, no_update, no_update
    s = jsonpickle.decode(browser_data)
    shape = s.shape

    s = BrowserState()
    G = Grid(s.shape)
    D = Holes(s.shape)

    s.xmax, s.ymax, s.zmax = shape[0], shape[1], shape[2]
    s.shape = shape

    instructions = process_string(input_string)

    instructions = [
        (get_node_index(*coords, s.shape), basis) for coords, basis in instructions
    ]

    print(f"Instructions: {instructions}")

    for i, measurementChoice in instructions:
        s.removed_nodes[i] = True
        G.handle_measurements(i, measurementChoice)
        s.log.append(f"{get_node_coords(i, s.shape)}, {measurementChoice}; ")
        s.log.append(html.Br())
        s.move_list.append([get_node_coords(i, s.shape), measurementChoice])
    return s.log, 1, "Graph loaded!", jsonpickle.encode(s), G.encode(), D.encode()


@callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("undo", "n_clicks"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("holes-data", "data"),
    prevent_initial_call=True,
)
def undo_move(n_clicks, browser_data, graphData, holeData):
    s = jsonpickle.decode(browser_data)

    if s.move_list:
        # Soft reset
        G = Grid(s.shape)
        D = Holes(s.shape, json_data=holeData)
        s.removed_nodes = np.zeros(s.xmax * s.ymax * s.zmax, dtype=bool)
        s.log = []

        undo = s.move_list.pop(-1)
        print(f"Undo: {undo}")
        for move in s.move_list:
            coords, measurementChoice = move
            i = get_node_index(*coords, s.shape)
            s.removed_nodes[i] = True
            G.handle_measurements(i, measurementChoice)
            s.log.append(f"{coords}, {measurementChoice}; ")
            s.log.append(html.Br())
        return s.log, 1, f"Undo: {undo}", jsonpickle.encode(s), G.encode(), D.encode()
    else:
        return (
            no_update,
            no_update,
            "Undo: No move to undo.",
            no_update,
            no_update,
            no_update,
        )
