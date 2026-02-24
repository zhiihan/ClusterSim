from textwrap import dedent as d
from dash import dcc, html, callback, Input, Output, State, no_update
from cluster_sim.app import BrowserState, grid_graph_3d, layouts
from cluster_sim.simulator import ClusterState
import re

import dash_bootstrap_components as dbc

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
            dcc.Store(id="graph-data"),
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
        r"\[(?P<x>\d+) \s?(?P<y>\d+) \s?(?P<z>\d+)\],\s?(?P<basis>[XYZ]);\s?"
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
    Input("load-graph-button", "n_clicks"),
    State("load-graph-input", "value"),
    State("browser-data", "data"),
    prevent_initial_call=True,
)
def load_graph_from_string(n_clicks, input_string, browser_data):

    if input_string is None:
        return no_update, no_update, no_update, no_update, no_update

    browser_state = BrowserState.from_json(browser_data)

    G = ClusterState.from_rustworkx(grid_graph_3d(browser_state.shape))
    layout = layouts[browser_state.layout](
        browser_state=browser_state
    )

    instructions = process_string(input_string)
    instructions = [
        (layout.get_node_index(*coords), basis) for coords, basis in instructions
    ]

    browser_state.log = ""
    browser_state.removed_nodes = set()
    print(f"Instructions: {instructions}")

    for i, measurementChoice in instructions:
        browser_state.removed_nodes.add(i)
        G.measure(i, measurementChoice)
        browser_state.log += f"{layout.get_node_coords(i)}, {measurementChoice};\n"
    return browser_state.log, 1, "Graph loaded!", browser_state.to_json(), G.to_json()


@callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Input("undo", "n_clicks"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    prevent_initial_call=True,
)
def undo_move(n_clicks, browser_data, graphData):
    browser_state = BrowserState.from_json(browser_data)

    if not browser_state.log:
        return (
            no_update,
            no_update,
            "Undo: No move to undo.",
            no_update,
            no_update,
        )

    move_list = browser_state.log.replace("\n", "").split(";")[:-1]
    undo = move_list.pop(-1)
    input_string = ";".join(move_list) + ";"

    log, _, ui, browser_statejson, gjson = load_graph_from_string(
        n_clicks, input_string, browser_data
    )
    return log, 1, f"Undo: {undo}", browser_statejson, gjson
