from cluster_sim.app.layout import update_plot_from_simulator

from cluster_sim import ClusterState
from cluster_sim.app import BrowserState, layouts, rx_graph_to_plot, update_plot_plotly
from textwrap import dedent as d
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objects as go
import rustworkx as rx
import dash_bootstrap_components as dbc
from .holes_rx import find_lattice, find_max_connected_lattice, build_centers_graph, connected_cube_to_nodes

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
                    dbc.Button("Find Percolation", id="percolation"),
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
            dcc.Store('holes-data')
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
    Output("figure-app", "figure", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Input("percolation", "n_clicks"),
    State("figure-app", "relayoutData"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("select-cubes", "value"),
    prevent_initial_call=True,
)
def find_unit_cells(
    nclicks, relayoutData, browser_data, graphData, select_cubes,
):

    browser_state = BrowserState.from_json(browser_data)
    G = ClusterState.from_json(graphData)
    layout = layouts[browser_state.layout](
        browser_state=browser_state
    )

    
    cubes = find_lattice(layout, browser_state.removed_nodes)
    click_number = 0
    # click_number = nclicks % (len(browser_state.valid_unit_cells))

    if select_cubes == "Select One Cube":
        pass
    elif select_cubes == "Select All Cubes":
        C = build_centers_graph(cubes, layout)
        connected_cubes = [C.subgraph(list(c)) for c in rx.connected_components(C)]

        click_number = nclicks % (len(connected_cubes))
        X = connected_cube_to_nodes(connected_cubes[click_number])

        ui = f"FindCluster: Displaying {click_number+1}/{len(connected_cubes)}"
    elif select_cubes == "Select All Connected Cubes":
        pass 

    nodes, edges, _ = rx_graph_to_plot(X, browser_state=browser_state)

    lattice = go.Scatter3d(
        x=nodes[:, 0],
        y=nodes[:, 1],
        z=nodes[:, 2],
        mode="markers",
        line=dict(color="blue", width=2),
        hoverinfo="none",
    )

    lattice_edges = go.Scatter3d(
        x=edges[:, 0],
        y=edges[:, 1],
        z=edges[:, 2],
        mode="lines",
        line=dict(color="blue", width=2),
        hoverinfo="none",
    )

    # Perform a manual update
    plotdata = update_plot_from_simulator(G, browser_state)

    plotdata += [lattice, lattice_edges]

    fig = update_plot_plotly(plotdata, browser_state)

    return browser_state.log, fig, "placeholder", browser_state.to_json(), G.to_json()
