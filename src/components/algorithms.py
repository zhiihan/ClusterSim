from textwrap import dedent as d
from dash import dcc, html, callback, Input, Output, State, no_update
import jsonpickle
import numpy as np
from cluster_sim.app import (
    get_node_index,
    get_node_coords,
    nx_to_plot,
    taxicab_metric,
)
import plotly.graph_objects as go
import networkx as nx
import itertools
import dash_bootstrap_components as dbc
import logging
from cluster_sim.simulator import ClusterState, NetworkXState


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
                    # dbc.Button("Find Lattice", id="findlattice"),
                    # dbc.Button("Find Cluster", id="alg2"),
                    # dbc.Button("Repair Grid", id="repair"),
                    dbc.Button("Reduction", id="reduction"),
                    # dbc.Button("Find Cluster v2", id="findlattice2"),
                ],
                direction="horizontal",
                gap=3,
            ),
            html.Hr(),
            html.B("Scale Factor"),
            dcc.Slider(
                id="rhg-slider",
                min=1,
                max=5,
                step=1,
                value=1,
                tooltip={
                    "placement": "bottom",
                },
                className="dash-bootstrap",
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
    G = ClusterState.from_json(graphData)
    D = NetworkXState.from_json(holeData)

    holes = D.graph.nodes

    hole_locations = np.zeros(
        (scale_factor + 1, scale_factor + 1, scale_factor + 1), dtype=int
    )

    # Finding the offset that maximizes holes placed in hole locations
    for h in holes:
        x, y, z = get_node_coords(h, shape=s.shape)

        x_vec = np.array([x, y, z])
        for xoffset, yoffset, zoffset in itertools.product(
            range(scale_factor + 1), repeat=3
        ):
            test_cond = x_vec % (scale_factor + 1)
            offset = np.array([xoffset, yoffset, zoffset])

            if np.all(test_cond == offset) or np.all(test_cond != offset):
                hole_locations[tuple(offset)] += 1

    logging.info(f"Hole locations:\n{np.ravel(hole_locations)}")

    # Finding the offset that maximizes holes placed in hole locations
    # Can use other indices to find other maximizing offsets
    s.offset = np.argwhere(hole_locations == np.max(hole_locations))[0]

    # Measuring the qubits based on the offset
    for z in range(s.shape[2]):
        for y in range(s.shape[1]):
            for x in range(s.shape[0]):
                x_vec = np.array([x, y, z]) % (scale_factor + 1)

                offset = np.array(s.offset)

                if np.all(x_vec == offset) or np.all(x_vec != offset):
                    i = get_node_index(x, y, z, s.shape)
                    if not s.removed_nodes[i]:
                        G.measure(i, "Z")
                        s.log.append(f"{get_node_coords(i, s.shape)}, Z; ")
                        s.log.append(html.Br())
                        s.removed_nodes[i] = True
                        s.move_list.append([get_node_coords(i, s.shape), "Z"])

    xoffset, yoffset, zoffset = s.offset

    # s.cubes, s.n_cubes = D.find_lattice(s.removed_nodes, s.offset)
    ui = f"Applied Algorithm: RHG Lattice, scale_factor = {scale_factor}, offset = {s.offset}"

    s.scale_factor = scale_factor

    return s.log, 1, ui, s.to_json(), G.to_json(), D.to_json()


@callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("reduction", "n_clicks"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("holes-data", "data"),
    State("select-cubes", "value"),
    prevent_initial_call=True,
)
def reduce_lattice(
    nclicks, browser_data, graphData, holeData, select_cubes, algorithm="line"
):

    s = jsonpickle.decode(browser_data)
    G = ClusterState.from_json(graphData)
    D = NetworkXState.from_json(holeData)

    if getattr(s, "scale_factor", None) is None:
        ui = "Find Cluster: Run RHG Lattice first."
        return no_update, no_update, ui, no_update, no_update, no_update

    s.valid_unit_cells, s.unit_cell_shape = generate_unit_cell_global_coords(
        s.shape, s.scale_factor, s.offset
    )

    if not s.valid_unit_cells:
        return (
            no_update,
            no_update,
            "No valid unit cells found.",
            s.to_json(),
            G.to_json(),
            D.to_json(),
        )

    click_number = nclicks % (len(s.valid_unit_cells))

    if select_cubes == "Select One Cube":
        H, imperfection_score = find_rings(
            G,
            s.scale_factor,
            shape=s.shape,
            unit_cell_coord=s.valid_unit_cells[click_number],
        )
        ui = f"Reduction: Imperfection score = {imperfection_score}"

    elif select_cubes == "Select All Cubes":
        graphs = []
        imperfection_score = 0
        for unit_cell_coord in s.valid_unit_cells:
            H, _ = find_rings(
                G,
                s.scale_factor,
                shape=s.shape,
                unit_cell_coord=unit_cell_coord,
            )
            graphs.append(H)
        H = nx.compose_all(graphs)

        imperfection_score = nx.number_of_isolates(H)

        logging.info(f"Total Imperfection score: {imperfection_score}")

        ui = f"Reduction: Total Imperfection score = {imperfection_score}, Equivalent 1-cell deletion ratio = {100 * imperfection_score / H.number_of_nodes()}%"
    elif select_cubes == "Select All Connected Cubes":

        C = nx.Graph()
        graphs_hashmap = {}
        imperfect_cells = 0

        for unit_cell_coord in s.valid_unit_cells:
            H_subgraph, imperfection_score = find_rings(
                G, s.scale_factor, shape=s.shape, unit_cell_coord=unit_cell_coord
            )
            if imperfection_score == 0:
                C.add_node(tuple(unit_cell_coord))
                graphs_hashmap[tuple(unit_cell_coord)] = H_subgraph

                for c in C.nodes:
                    if taxicab_metric(np.array(c), unit_cell_coord) <= (
                        s.scale_factor + 1
                    ):
                        C.add_edge(c, tuple(unit_cell_coord))
            else:
                imperfect_cells += 1

        connected_clusters = [C.subgraph(c).copy() for c in nx.connected_components(C)]

        if not connected_clusters:
            ui = "Reduction: No connected clusters found."
            s.lattice = None
            s.lattice_edges = None
            return s.log, 1, ui, s.to_json(), G.to_json(), D.to_json()

        connected_cluster_graph = connected_clusters[nclicks % len(connected_clusters)]

        graphs = [
            graphs_hashmap[unit_cell_coord]
            for unit_cell_coord in connected_cluster_graph.nodes
        ]

        H = nx.compose_all(graphs)

        imperfection_score = nx.number_of_isolates(H)
        perfect_cells = len(graphs)

        logging.info(
            f"Finding connected components: {perfect_cells} perfect cells, {imperfect_cells} imperfect cells."
        )

        ui = f"Reduction: Displaying cluster {nclicks % len(connected_clusters) + 1}/{len(connected_clusters)}, perfect cells = {perfect_cells}, imperfect cells = {imperfect_cells}"

    nodes, edges = nx_to_plot(H, shape=s.shape, index=True)

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

    return s.log, 1, ui, s.to_json(), G.to_json(), D.to_json()


def generate_ring(scale_factor, global_offset, j, ring_gen_funcs):
    """
    Generate the rings of a unit cell in a Raussendorf lattice.
    """

    ring_node_coords = set()
    for i in range(0, scale_factor + 2):
        for f in ring_gen_funcs:
            ring_node_coords.add(tuple(np.array(global_offset) + np.array(f(i, j))))
    return list(ring_node_coords)


def find_rings(G, scale_factor, shape, unit_cell_coord=(0, 0, 0)):
    """
    Find the rings of a unit cell in a Raussendorf lattice.
    """

    unit_cell_coord = np.array(unit_cell_coord)

    ring_gen_funcs_x = [
        lambda i, j: (j, 0, i),
        lambda i, j: (j, i, scale_factor + 1),
        lambda i, j: (j, scale_factor + 1, i),
        lambda i, j: (j, i, 0),
    ]

    ring_gen_funcs_y = [
        lambda i, j: (0, j, i),
        lambda i, j: (i, j, scale_factor + 1),
        lambda i, j: (scale_factor + 1, j, i),
        lambda i, j: (i, j, 0),
    ]

    ring_gen_funcs_z = [
        lambda i, j: (0, i, j),
        lambda i, j: (i, scale_factor + 1, j),
        lambda i, j: (scale_factor + 1, i, j),
        lambda i, j: (i, 0, j),
    ]

    optimized_rings = []
    imperfection_score = 0

    for ring_gen in [ring_gen_funcs_x, ring_gen_funcs_y, ring_gen_funcs_z]:
        rings = {}
        for j in range(1, scale_factor + 1):
            ring_list = generate_ring(scale_factor, unit_cell_coord, j, ring_gen)
            counter = evaluate_ring(G, ring_list)

            if counter == 0:
                optimized_rings.append(ring_list)
                logging.info(f"Ring {j} is perfect, no erasures found.")
                break
            else:
                rings[j] = counter
                logging.info(f"Ring {j} has {counter} erasures.")
        else:
            best_j = min(rings, key=rings.get)
            logging.info(f"Best ring is {best_j} with {rings[best_j]} erasures.")

            ring_list = generate_ring(scale_factor, unit_cell_coord, best_j, ring_gen)
            optimized_rings.append(ring_list)
            imperfection_score += rings[best_j]

    optimized_rings = [item for sublist in optimized_rings for item in sublist]

    optimized_rings = [get_node_index(*item, shape=shape) for item in optimized_rings]

    return G.graph.subgraph(optimized_rings), imperfection_score


def evaluate_ring(G, ring_list):
    """
    Evaluate the number of nodes in a ring that are not in the graph.
    This is used to determine how many nodes are missing from the graph.
    """
    counter = 0
    for node in ring_list:
        if nx.is_isolate(G.graph, node):
            counter += 1
    return counter


def generate_unit_cell_global_coords(shape, scale_factor, offset) -> list[np.ndarray]:
    """
    Find the bottom left corner of each unit cell in a 3D grid.
    """

    # Calculate the number of cubes in each dimension
    num_cubes = np.array(shape) // (scale_factor + 1)

    unit_cell_locations = []

    unit_cell_shape = np.array([0, 0, 0], dtype=int)
    for i in itertools.product(
        range(num_cubes[0]), range(num_cubes[1]), range(num_cubes[2])
    ):
        # print(f"Unit cell location: {i}, scale factor: {scale_factor}")

        if any(
            np.array(i) * (scale_factor + 1) + (scale_factor + 2) + offset
            > np.array(shape)
        ):
            # print(f"Skipping unit cell {i} as it exceeds grid dimensions.")
            continue

        unit_cell_locations.append(np.array(i) * (scale_factor + 1) + offset)

        unit_cell_shape = np.maximum(np.array(i), unit_cell_shape)

    return unit_cell_locations, unit_cell_shape + np.array([1, 1, 1])
