import plotly.graph_objects as go
from cluster_sim.app.grid import Grid
from cluster_sim.app.holes import Holes
from cluster_sim.app.state import BrowserState
from cluster_sim.app.utils import (
    get_node_index,
    get_node_coords,
    path_to_plot,
    nx_to_plot,
)
import dash
from dash import html, Input, Output, State
import time
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from dash_resizable_panels import PanelGroup, Panel, PanelResizeHandle
import numpy as np
import networkx as nx
from components import (
    move_log,
    reset_graph,
    algorithms,
    hover_data,
    zoom_data,
    load_graph,
    measurementbasis,
    display_options,
    figure,
    error_channel,
)

jsonpickle_numpy.register_handlers()

app = dash.Dash(
    __name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"]
)
app.title = "Cluster Sim"
server = app.server  # For deployment

app.layout = html.Div(
    [
        PanelGroup(
            id="main_app",
            children=[
                Panel(
                    id="resize_figure",
                    children=[
                        figure,
                    ],
                ),
                PanelResizeHandle(
                    html.Div(
                        style={
                            "backgroundColor": "grey",
                            "height": "100%",
                            "width": "5px",
                        }
                    )
                ),
                Panel(
                    id="resize_info",
                    children=[
                        html.Div(
                            [
                                algorithms,
                                move_log,
                            ],
                            className="four columns",
                        ),
                        html.Div(
                            [
                                html.Div(id="ui"),
                                measurementbasis,
                                display_options,
                                hover_data,
                                zoom_data,
                            ],
                            className="four columns",
                        ),
                        html.Div(
                            [
                                reset_graph,
                                error_channel,
                                load_graph,
                            ],
                            className="four columns",
                        ),
                    ],
                    style={"overflow": "scroll", "width": "95%"},
                ),
            ],
            direction="horizontal",
            style={"height": "95vh"},
        )
    ]
)


@app.callback(
    Output("browser-data", "data"),
    Output("graph-data", "data"),
    Output("holes-data", "data"),
    Input("none", "children"),
)
def initial_call(dummy):
    """
    Initialize the graph in the browser as a JSON object.
    """
    s = BrowserState()
    G = Grid(s.shape)
    D = Holes(s.shape)

    return jsonpickle.encode(s), G.encode(), D.encode()


@app.callback(
    Output("click-data", "children"),
    Output("draw-plot", "data"),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("basic-interactions", "clickData"),
    State("radio-items", "value"),
    State("click-data", "children"),
    State("basic-interactions", "hoverData"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("holes-data", "data"),
    prevent_initial_call=True,
)
def display_click_data(
    clickData, measurementChoice, clickLog, hoverData, browser_data, graphData, holeData
):
    """
    Updates the browser state if there is a click.
    """
    if not clickData:
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )
    point = clickData["points"][0]

    # Do nothing if clicked on edges
    if point["curveNumber"] > 0 or "x" not in point:
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )
    else:
        s = jsonpickle.decode(browser_data)
        G = Grid(s.shape, json_data=graphData)
        D = Holes(s.shape, json_data=holeData)

        i = get_node_index(point["x"], point["y"], point["z"], s.shape)
        # Update the plot based on the node clicked
        if measurementChoice == "Z:Hole":
            D.add_node(i)
            measurementChoice = "Z"  # Handle it as if it was Z measurement
        if s.removed_nodes[i] == False:
            s.removed_nodes[i] = True
            G.handle_measurements(i, measurementChoice)
            s.move_list.append([i, measurementChoice])
            ui = f"Clicked on {i} at {get_node_coords(i, s.shape)}"
        s.log.append(f"{i}, {measurementChoice}; ")
        s.log.append(html.Br())

        # This solves the double click issue
        time.sleep(0.1)
        return html.P(s.log), i, ui, jsonpickle.encode(s), G.encode(), D.encode()


@app.callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("alg1", "n_clicks"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("holes-data", "data"),
    prevent_initial_call=True,
)
def algorithm1(nclicks, browser_data, graphData, holeData):
    """
    Create a RHG lattice from a square lattice.
    """
    s = jsonpickle.decode(browser_data)
    G = Grid(s.shape, json_data=graphData)
    D = Holes(s.shape, json_data=holeData)

    holes = D.graph.nodes
    hole_locations = np.zeros(8)
    xoffset, yoffset, zoffset = s.offset

    # counting where the holes are
    for h in holes:
        x, y, z = h
        for zoffset in range(2):
            for yoffset in range(2):
                for xoffset in range(2):
                    if ((x + xoffset) % 2 == (z + zoffset) % 2) and (
                        (y + yoffset) % 2 == (z + zoffset) % 2
                    ):
                        hole_locations[xoffset + yoffset * 2 + zoffset * 4] += 1

    print(hole_locations)

    xoffset = np.argmax(hole_locations) % 2
    yoffset = np.argmax(hole_locations) // 2
    zoffset = np.argmax(hole_locations) // 4

    s.offset = [xoffset, yoffset, zoffset]

    print(f"xoffset, yoffset, zoffset = {(xoffset, yoffset, zoffset)}")

    for z in range(s.shape[2]):
        for y in range(s.shape[1]):
            for x in range(s.shape[0]):
                if ((x + xoffset) % 2 == (z + zoffset) % 2) and (
                    (y + yoffset) % 2 == (z + zoffset) % 2
                ):
                    i = get_node_index(x, y, z, s.shape)
                    if s.removed_nodes[i] == False:
                        G.handle_measurements(i, "Z")
                        s.log.append(f"{i}, Z; ")
                        s.log.append(html.Br())
                        s.removed_nodes[i] = True
                        s.move_list.append([i, "Z"])

    s.cubes, s.n_cubes = D.find_lattice(s.removed_nodes, xoffset, yoffset, zoffset)
    ui = f"RHG: Created RHG Lattice."

    return s.log, 1, ui, jsonpickle.encode(s), G.encode(), D.encode()


@app.callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("findlattice", "n_clicks"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("holes-data", "data"),
    prevent_initial_call=True,
)
def find_lattice(nclicks, browser_data, graphData, holeData):
    """
    Returns:
    """
    s = jsonpickle.decode(browser_data)
    G = Grid(s.shape, json_data=graphData)
    D = Holes(s.shape, json_data=holeData)

    try:
        if s.offset[0] == None:
            # cubes, n_cubes is not defined and this is because we didnt compute the offsets.
            ui = "FindLattice: Run RHG Lattice first."
            return s.log, 1, ui, jsonpickle.encode(s), G.encode(), D.encode()

        if s.n_cubes is None:
            s.cubes, s.n_cubes = D.find_lattice(
                s.removed_nodes, s.xoffset, s.yoffset, s.zoffset
            )

        click_number = nclicks % (len(s.cubes))

        if len(s.cubes) > 0:
            C = nx.Graph()
            C.add_node(tuple(s.cubes[click_number][0, :]))

            X = D.connected_cube_to_nodes(C)

            nodes, edges = nx_to_plot(X, shape=s.shape, index=False)

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
            ui = f"FindLattice: Displaying {click_number+1}/{len(s.cubes)} unit cells found for p = {s.p}, shape = {s.shape}"

            s.lattice = lattice.to_json()
            s.lattice_edges = lattice_edges.to_json()
    except NameError:
        # cubes, n_cubes is not defined and this is because we didnt compute the offsets.
        ui = "FindLattice: Run RHG Lattice first."
    return (
        s.log,
        1,
        ui,
        jsonpickle.encode(s),
        G.encode(),
        D.encode(),
    )


@app.callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("alg2", "n_clicks"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("holes-data", "data"),
    prevent_initial_call=True,
)
def find_cluster(nclicks, browser_data, graphData, holeData):
    """
    Find a cluster of connected cubes in the lattice.
    """
    s = jsonpickle.decode(browser_data)
    G = Grid(s.shape, json_data=graphData)
    D = Holes(s.shape, json_data=holeData)

    try:
        if s.offset[0] == None:
            # cubes, n_cubes is not defined and this is because we didnt compute the offsets.
            ui = "FindLattice: Run RHG Lattice first."
            return s.log, 1, ui, jsonpickle.encode(s), G.encode(), D.encode()

        C = D.build_centers_graph(s.cubes)
        connected_cubes = D.find_connected_lattice(C)
        for i in connected_cubes:
            print(i, len(connected_cubes))

        if len(connected_cubes) > 0:
            click_number = nclicks % (len(connected_cubes))
            X = D.connected_cube_to_nodes(connected_cubes[click_number])

            nodes, edges = nx_to_plot(X, shape=s.shape, index=False)

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
            ui = f"FindCluster: Displaying {click_number+1}/{len(connected_cubes)}, unit cells = {len(connected_cubes[click_number].nodes)}, edges = {len(connected_cubes[click_number].edges)}"
        else:
            ui = f"FindCluster: No cubes found."
    except TypeError:
        ui = "FindCluster: Run RHG Lattice first."
    except NameError:
        ui = "FindCluster: Run RHG Lattice first."
    return s.log, 2, ui, jsonpickle.encode(s), G.encode(), D.encode()


@app.callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("alg3", "n_clicks"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("holes-data", "data"),
    prevent_initial_call=True,
)
def find_percolation(nclicks, browser_data, graphData, holeData):
    """
    Find a path from the top of the grid to the bottom of the grid.
    """
    s = jsonpickle.decode(browser_data)
    G = Grid(s.shape, json_data=graphData)
    D = Holes(s.shape, json_data=holeData)

    gnx = G.to_networkx()

    removed_nodes_reshape = s.removed_nodes.reshape((s.xmax, s.ymax, s.zmax))

    zeroplane = removed_nodes_reshape[:, :, 0]
    zmaxplane = removed_nodes_reshape[:, :, s.zmax - 1]

    x = np.argwhere(
        zeroplane == 0
    )  # This is the coordinates of all valid node in z = 0
    y = np.argwhere(
        zmaxplane == 0
    )  # This is the coordinates of all valid node in z = L

    path = None
    while path is None:
        try:
            i = get_node_index(*x[s.path_clicks % len(x)], 0, s.shape)
            j = get_node_index(*y[s.path_clicks // len(x)], s.zmax - 1, s.shape)
            path = nx.shortest_path(gnx, i, j)
        except nx.exception.NetworkXNoPath:
            ui = "No path."
            print(f"no path, {i}, {j}")
        finally:
            s.path_clicks += 1

    nodes, edges = path_to_plot(path, s.shape)

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

    ui = f"Found percolation from z = 0, {get_node_coords(i, shape=s.shape)} to z = {s.zmax}, {get_node_coords(j, shape=s.shape)}"
    return s.log, 1, ui, jsonpickle.encode(s), G.encode(), D.encode()


@app.callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("repair", "n_clicks"),
    State("browser-data", "data"),
    State("holes-data", "data"),
    prevent_initial_call=True,
)
def repair_grid(nclicks, browser_data, holeData):
    s = jsonpickle.decode(browser_data)
    D = Holes(s.shape, json_data=holeData)

    repairs, failures = D.repair_grid(s.p)

    G = Grid(s.shape)
    s.removed_nodes = np.zeros(s.xmax * s.ymax * s.zmax, dtype=bool)
    s.log = []
    for f in failures:
        i = get_node_index(*f, s.shape)
        s.removed_nodes[i] = True
        G.handle_measurements(i, "Z")
        s.log.append(f"{i}, Z; ")
        s.log.append(html.Br())
        s.move_list.append([i, "Z"])

    if len(repairs) + len(failures) > 0:
        rate = len(repairs) / (len(repairs) + len(failures))
        ui = f"Repairs = {len(repairs)}, Failures = {len(failures)} Repair Rate = {rate:.2f}, Holes = {np.sum(s.removed_nodes)}, peff={np.sum(s.removed_nodes)/(s.xmax*s.ymax*s.zmax)}"
    else:
        ui = "All qubits repaired!"
    return s.log, 1, ui, jsonpickle.encode(s), G.encode(), D.encode()


if __name__ == "__main__":
    app.run(debug=True)
