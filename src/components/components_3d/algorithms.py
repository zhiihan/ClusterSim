from networkx.drawing.nx_agraph import from_agraph
from cluster_sim import ClusterState
from cluster_sim.app import BrowserState, layouts, rx_graph_to_plot
from textwrap import dedent as d
from dash import dcc, html, callback, Input, Output, State, no_update
import numpy as np
import plotly.graph_objects as go
import rustworkx as rx
import itertools
import dash_bootstrap_components as dbc
import logging


algorithms = dbc.Card(
    dbc.CardBody(
        [
            dcc.Markdown(
                d(
                    """
                    **Algorithms**

                    Click on points in the graph.
                """
                )
            ),
            dbc.Stack(
                [
                    dbc.Button("RHG Lattice", id="alg1"),
                    dbc.Button("Find Percolation", id="Percolation"),
                ],
                direction="horizontal",
                gap=3,
            ),
            html.Hr(),
            dcc.Dropdown(
                ["Select One Cube", "Select All Cubes", "Select All Connected Cubes"],
                "Select All Cubes",
                id="select-cubes",
                className="dash-bootstrap",
            ),
        ],
    )
)


@callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Input("alg1", "n_clicks"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    prevent_initial_call=True,
)
def rhg_lattice_scale(nclicks, browser_data, graphData):
    """
    Create a RHG lattice from a square lattice, with a scale factor.

    Parameters:
    - nclicks: number of clicks on the button (unused)
    - scale_factor: scale factor for the RHG lattice
    - browser_data: data from the browser
    - graphData: data from the graph
    - holeData: data from the holes
    """
    browser_state = BrowserState.from_json(browser_data)
    G = ClusterState.from_json(graphData)

    layout = layouts[browser_state.layout](
        browser_state=browser_state
    )

    for node_index in range(len(G)):
        coords = layout.get_node_coords(node_index)
        if (coords[0] % 2) == (coords[1] % 2) == (coords[2] % 2):
            G.measure(node_index, force=0, basis='Z')
            browser_state.log += f"{layout.get_node_coords(node_index)}, Z;\n"
            browser_state.removed_nodes.add(node_index)

    ui = "Created an RHG lattice."

    return browser_state.log, 1, ui, browser_state.to_json(), G.to_json()


@callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("percolation", "n_clicks"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("select-cubes", "value"),
    prevent_initial_call=True,
)
def find_unit_cells(
    nclicks, browser_data, graphData, select_cubes,
):

    browser_state = BrowserState.from_json(browser_data)
    G = ClusterState.from_json(graphData)

    click_number = nclicks % (len(browser_state.valid_unit_cells))

    generate_unit_cell_global_coords(browser_state.shape)

    H = rx.PyGraph()

    if select_cubes == "Select One Cube":
        
    elif select_cubes == "Select All Cubes":
        pass
    elif select_cubes == "Select All Connected Cubes":
        pass 

    nodes, edges = rx_graph_to_plot(H, browser_state=)

    lattice = go.Scatter3d(
        x=nodes[0],
        y=nodes[1],
        z=nodes[2],
        mode="markers",
        line=dict(color="blue", width=2),
        hoverinfo="none",
    )

    lattice_edges = go.Scatter3d(
        x=edges[0],
        y=edges[1],
        z=edges[2],
        mode="lines",
        line=dict(color="blue", width=2),
        hoverinfo="none",
    )

    s.lattice = lattice.to_json()
    s.lattice_edges = lattice_edges.to_json()

    return s.log, 1, ui, jsonpickle.encode(s), G.encode(), D.encode()


def generate_unit_cell_global_coords(shape) -> list[np.ndarray]:
    """
    Find the bottom left corner of each unit cell in a 3D grid.
    """

    num_cubes = np.array(shape) // 2
    unit_cell_locations = []

    unit_cell_shape = np.array([0, 0, 0], dtype=int)
    for i in itertools.product(
        range(num_cubes[0]), range(num_cubes[1]), range(num_cubes[2])
    ):
        # print(f"Unit cell location: {i}, scale factor: {scale_factor}")

        if any(
            np.array(i) * 2 + 3
            > np.array(shape)
        ):
            # print(f"Skipping unit cell {i} as it exceeds grid dimensions.")
            continue

        unit_cell_locations.append(np.array(i) * 2)

    return unit_cell_locations


def unit_cell_check():
