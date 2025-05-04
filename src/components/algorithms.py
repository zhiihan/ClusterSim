from textwrap import dedent as d
from dash import dcc, html, callback, Input, Output, State, no_update
import jsonpickle
import numpy as np
from cluster_sim.app.utils import get_node_index
from cluster_sim.app.grid import Grid
from cluster_sim.app.holes import Holes
from cluster_sim.app.utils import (
    get_node_index,
    get_node_coords,
    path_to_plot,
    nx_to_plot,
)
import plotly.graph_objects as go
import networkx as nx
import itertools

algorithms = html.Div(
    [
        dcc.Markdown(
            d(
                """
                    **Algorithms**

                    Click on points in the graph.
                """
            )
        ),
        html.Button("RHG Lattice", id="alg1"),
        html.Button("Find Lattice", id="findlattice"),
        html.Button("Find Cluster", id="alg2"),
        html.Button("Repair Grid", id="repair"),
        html.Button("Find Cluster v2", id="findlattice2"),
        html.P("Scale Factor"),
        dcc.Slider(
            id="rhg-slider",
            min=1,
            max=3,
            step=1,
            value=1,
            tooltip={
                "placement": "bottom",
                "always_visible": True,
            },
        ),
    ],
)


@callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("alg1", "n_clicks"),
    State("rhg-slider", "value"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("holes-data", "data"),
    prevent_initial_call=True,
)
def rhg_lattice_scale(nclicks, scale_factor, browser_data, graphData, holeData):
    """
    Create a RHG lattice from a square lattice, with a scale factor.

    Parameters:
    - nclicks: number of clicks on the button (unused)
    - scale_factor: scale factor for the RHG lattice
    - browser_data: data from the browser
    - graphData: data from the graph
    - holeData: data from the holes
    """
    s = jsonpickle.decode(browser_data)
    G = Grid(s.shape, json_data=graphData)
    D = Holes(s.shape, json_data=holeData)

    holes = D.graph.nodes

    hole_locations = np.zeros((scale_factor + 1) ** 3, dtype=int)

    # counting where the holes are
    for h in holes:
        x, y, z = h

        x_vec = np.array([x, y, z])
        for xoffset, yoffset, zoffset in itertools.product(
            range(scale_factor + 1), repeat=3
        ):
            test_cond = x_vec % (scale_factor + 1)
            offset = np.array([xoffset, yoffset, zoffset])

            if np.all(test_cond == offset) or np.all(test_cond != offset):
                hole_locations[
                    xoffset
                    + yoffset * (scale_factor + 1)
                    + zoffset * (scale_factor + 1) ** 2
                ] += 1

    print("hole locations", hole_locations)

    xoffset = np.argmax(hole_locations) % (scale_factor + 1)
    yoffset = np.argmax(hole_locations) // (scale_factor + 1)
    zoffset = np.argmax(hole_locations) // (scale_factor + 1) ** 2

    s.offset = [int(xoffset), int(yoffset), int(zoffset)]

    for z in range(s.shape[2]):
        for y in range(s.shape[1]):
            for x in range(s.shape[0]):
                x_vec = (np.array([x, y, z])) % (scale_factor + 1)
                offset = np.array([xoffset, yoffset, zoffset])
                if np.all(x_vec == offset) or np.all(x_vec != offset):
                    i = get_node_index(x, y, z, s.shape)
                    if s.removed_nodes[i] == False:
                        G.handle_measurements(i, "Z")
                        s.log.append(f"{i}, Z; ")
                        s.log.append(html.Br())
                        s.removed_nodes[i] = True
                        s.move_list.append([i, "Z"])

    s.cubes, s.n_cubes = D.find_lattice(s.removed_nodes, xoffset, yoffset, zoffset)
    ui = f"Applied Algorithm: RHG Lattice, scale_factor = {scale_factor}, offset = {s.offset}"

    s.scale_factor = scale_factor

    return s.log, 1, ui, jsonpickle.encode(s), G.encode(), D.encode()


# @callback(
#     Output("click-data", "children", allow_duplicate=True),
#     Output("draw-plot", "data", allow_duplicate=True),
#     Output("ui", "children", allow_duplicate=True),
#     Output("browser-data", "data", allow_duplicate=True),
#     Output("graph-data", "data", allow_duplicate=True),
#     Output("holes-data", "data", allow_duplicate=True),
#     Input("alg1", "n_clicks"),
#     State("browser-data", "data"),
#     State("graph-data", "data"),
#     State("holes-data", "data"),
#     prevent_initial_call=True,
# )
# def algorithm1(nclicks, browser_data, graphData, holeData):
#     """
#     Create a RHG lattice from a square lattice.
#     """
#     s = jsonpickle.decode(browser_data)
#     G = Grid(s.shape, json_data=graphData)
#     D = Holes(s.shape, json_data=holeData)

#     holes = D.graph.nodes
#     hole_locations = np.zeros(8)
#     xoffset, yoffset, zoffset = s.offset

#     # counting where the holes are
#     for h in holes:
#         x, y, z = h
#         for zoffset in range(2):
#             for yoffset in range(2):
#                 for xoffset in range(2):
#                     if ((x + xoffset) % 2 == (z + zoffset) % 2) and (
#                         (y + yoffset) % 2 == (z + zoffset) % 2
#                     ):
#                         hole_locations[xoffset + yoffset * 2 + zoffset * 4] += 1

#     print(hole_locations)

#     xoffset = np.argmax(hole_locations) % 2
#     yoffset = np.argmax(hole_locations) // 2
#     zoffset = np.argmax(hole_locations) // 4

#     s.offset = [xoffset, yoffset, zoffset]

#     print(f"xoffset, yoffset, zoffset = {(xoffset, yoffset, zoffset)}")

#     for z in range(s.shape[2]):
#         for y in range(s.shape[1]):
#             for x in range(s.shape[0]):
#                 if ((x + xoffset) % 2 == (z + zoffset) % 2) and (
#                     (y + yoffset) % 2 == (z + zoffset) % 2
#                 ):
#                     i = get_node_index(x, y, z, s.shape)
#                     if s.removed_nodes[i] == False:
#                         G.handle_measurements(i, "Z")
#                         s.log.append(f"{i}, Z; ")
#                         s.log.append(html.Br())
#                         s.removed_nodes[i] = True
#                         s.move_list.append([i, "Z"])

#     s.cubes, s.n_cubes = D.find_lattice(s.removed_nodes, xoffset, yoffset, zoffset)
#     ui = f"RHG: Created RHG Lattice."

#     return s.log, 1, ui, jsonpickle.encode(s), G.encode(), D.encode()


@callback(
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

    if getattr(s, "cubes", None) is None:
        return no_update, no_update, no_update, no_update, no_update, no_update

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


@callback(
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


# @callback(
#     Output("click-data", "children", allow_duplicate=True),
#     Output("draw-plot", "data", allow_duplicate=True),
#     Output("ui", "children", allow_duplicate=True),
#     Output("browser-data", "data", allow_duplicate=True),
#     Output("graph-data", "data", allow_duplicate=True),
#     Output("holes-data", "data", allow_duplicate=True),
#     Input("alg3", "n_clicks"),
#     State("browser-data", "data"),
#     State("graph-data", "data"),
#     State("holes-data", "data"),
#     prevent_initial_call=True,
# )
# def find_percolation(nclicks, browser_data, graphData, holeData):
#     """
#     Find a path from the top of the grid to the bottom of the grid.
#     """
#     s = jsonpickle.decode(browser_data)
#     G = Grid(s.shape, json_data=graphData)
#     D = Holes(s.shape, json_data=holeData)

#     gnx = G.to_networkx()

#     removed_nodes_reshape = s.removed_nodes.reshape((s.xmax, s.ymax, s.zmax))

#     zeroplane = removed_nodes_reshape[:, :, 0]
#     zmaxplane = removed_nodes_reshape[:, :, s.zmax - 1]

#     x = np.argwhere(
#         zeroplane == 0
#     )  # This is the coordinates of all valid node in z = 0
#     y = np.argwhere(
#         zmaxplane == 0
#     )  # This is the coordinates of all valid node in z = L

#     path = None
#     while path is None:
#         try:
#             i = get_node_index(*x[s.path_clicks % len(x)], 0, s.shape)
#             j = get_node_index(*y[s.path_clicks // len(x)], s.zmax - 1, s.shape)
#             path = nx.shortest_path(gnx, i, j)
#         except nx.exception.NetworkXNoPath:
#             ui = "No path."
#             print(f"no path, {i}, {j}")
#         finally:
#             s.path_clicks += 1

#     nodes, edges = path_to_plot(path, s.shape)

#     lattice = go.Scatter3d(
#         x=nodes[0],
#         y=nodes[1],
#         z=nodes[2],
#         mode="markers",
#         line=dict(color="blue", width=2),
#         hoverinfo="none",
#     )

#     lattice_edges = go.Scatter3d(
#         x=edges[0],
#         y=edges[1],
#         z=edges[2],
#         mode="lines",
#         line=dict(color="blue", width=2),
#         hoverinfo="none",
#     )

#     s.lattice = lattice.to_json()
#     s.lattice_edges = lattice_edges.to_json()

#     ui = f"Found percolation from z = 0, {get_node_coords(i, shape=s.shape)} to z = {s.zmax}, {get_node_coords(j, shape=s.shape)}"
#     return s.log, 1, ui, jsonpickle.encode(s), G.encode(), D.encode()


@callback(
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


@callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("findlattice2", "n_clicks"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("holes-data", "data"),
    prevent_initial_call=True,
)
def find_cluster2(nclicks, browser_data, graphData, holeData):
    """
    Find a cluster of connected cubes in the lattice.
    """
    s = jsonpickle.decode(browser_data)
    G = Grid(s.shape, json_data=graphData)
    D = Holes(s.shape, json_data=holeData)

    if getattr(s, "scale_factor", None) is None:
        return no_update, no_update, no_update, no_update, no_update, no_update

    ui = "wip"

    xoffset, yoffset, zoffset = s.offset

    print(f"offset = {s.offset}")

    H = check_graph_xy(G, xoffset=xoffset, yoffset=yoffset, zoffset=zoffset)

    nodes, edges = nx_to_plot(H, shape=s.shape, index=False)

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


def check_graph_xy(G, xoffset=0, yoffset=0, zoffset=0):
    """
    Check if a unit cell is a valid Raussendorf unit cell.
    A valid unit cell is a cube has 6 faces, each with 2 orientations, for a total of 12 oriented faces.

    If all oriented faces contains at least 1 line that does not contain an erasure, the unit cell is valid.
    """

    # fmt: off
    face1 = [[(xoffset + j, yoffset + 0, zoffset + d) for j in range(0, 4)] for d in range(1, 3)] 
    face2 = [[(xoffset + j, yoffset + d, zoffset + 0) for j in range(0, 4)] for d in range(1, 3)]
    face3 = [[(xoffset + j, yoffset + 3, zoffset + d) for j in range(0, 4)] for d in range(1, 3)]
    face4 = [[(xoffset + j, yoffset + d, zoffset + 3) for j in range(0, 4)] for d in range(1, 3)]
    face5 = [[(xoffset + d, yoffset + j, zoffset + 0) for j in range(0, 4)] for d in range(1, 3)]
    face6 = [[(xoffset + 0, yoffset + j, zoffset + d) for j in range(0, 4)] for d in range(1, 3)]
    face7 = [[(xoffset + d, yoffset + j, zoffset + 3) for j in range(0, 4)] for d in range(1, 3)]
    face8 = [[(xoffset + 3, yoffset + j, zoffset + d) for j in range(0, 4)] for d in range(1, 3)]
    face9 = [[(xoffset + d, yoffset + 3, zoffset + j) for j in range(0, 4)] for d in range(1, 3)]
    face10 = [[(xoffset + 3, yoffset + d, zoffset + j) for j in range(0, 4)] for d in range(1, 3)]
    face11 = [[(xoffset + d, yoffset + 0, zoffset + j) for j in range(0, 4)] for d in range(1, 3)]
    face12 = [[(xoffset + 0, yoffset + d, zoffset + j) for j in range(0, 4)] for d in range(1, 3)]

    all_faces = [face1, face2, face3, face4, face5, face6, face7, face8, face9, face10, face11, face12]
    # fmt: on

    all_faces_unzipped = [
        node for face in all_faces for checks in face for node in checks
    ]

    print(all_faces)

    H = G.graph.subgraph(all_faces_unzipped).copy()
    H.remove_nodes_from(list(nx.isolates(H)))

    print(H)

    joined_faces = []

    for face in all_faces:
        for checks in face:
            print(checks)
            if all(H.has_node(i) for i in checks):
                joined_faces.append(checks)
                break
        else:
            print("No face found")

            return G.graph.subgraph([node for l in face for node in l])  # debug

            return None

    joined_faces = [node for l in joined_faces for node in l]

    return G.graph.subgraph(joined_faces)
