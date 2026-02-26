import rustworkx as rx
import numpy as np

constant_cube = [
    np.array([0, -1, -1]),
    np.array([-1, 0, -1]),
    np.array([0, 0, -1]),
    np.array([0, 1, -1]),
    np.array([1, 0, -1]),
    np.array([-1, -1, 0]),
    np.array([0, -1, 0]),
    np.array([-1, 0, 0]),
    np.array([-1, 1, 0]),
    np.array([0, 1, 0]),
    np.array([1, 1, 0]),
    np.array([1, 0, 0]),
    np.array([1, -1, 0]),
    np.array([0, -1, 1]),
    np.array([-1, 0, 1]),
    np.array([0, 0, 1]),
    np.array([1, 0, 1]),
    np.array([0, 1, 1]),
]

taxicab2 = [
    np.array([-2, 0, 0], dtype=int),
    np.array([-1, -1, 0], dtype=int),
    np.array([-1, 0, -1], dtype=int),
    np.array([-1, 0, 1], dtype=int),
    np.array([-1, 1, 0], dtype=int),
    np.array([0, -2, 0], dtype=int),
    np.array([0, -1, -1], dtype=int),
    np.array([0, -1, 1], dtype=int),
    np.array([0, 0, -2], dtype=int),
    np.array([0, 1, -1], dtype=int),
    np.array([0, 1, 1], dtype=int),
    np.array([1, -1, 0], dtype=int),
    np.array([1, 0, -1], dtype=int),
    np.array([1, 0, 1], dtype=int),
    np.array([1, 1, 0], dtype=int),
]

taxicab3 = [
    np.array([-2, -1, 0], dtype=int),
    np.array([-2, 0, -1], dtype=int),
    np.array([-2, 0, 1], dtype=int),
    np.array([-2, 1, 0], dtype=int),
    np.array([-1, -2, 0], dtype=int),
    np.array([-1, -1, -1], dtype=int),
    np.array([-1, -1, 1], dtype=int),
    np.array([-1, 0, -2], dtype=int),
    np.array([-1, 1, -1], dtype=int),
    np.array([-1, 1, 1], dtype=int),
    np.array([0, -2, -1], dtype=int),
    np.array([0, -2, 1], dtype=int),
    np.array([0, -1, -2], dtype=int),
    np.array([0, 1, -2], dtype=int),
    np.array([1, -2, 0], dtype=int),
    np.array([1, -1, -1], dtype=int),
    np.array([1, -1, 1], dtype=int),
    np.array([1, 0, -2], dtype=int),
    np.array([1, 1, -1], dtype=int),
    np.array([1, 1, 1], dtype=int),
]

box = [
    np.array([1, 0, 0]),
    np.array([-1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, -1, 0]),
    np.array([0, 0, 1]),
    np.array([0, 0, -1]),
]


def find_lattice(layout, removed_nodes, max_scale=1):
    """
    Find a raussendorf lattice.

    Returns: cubes: a list containing a cube:

        cube: np.array with shape (19, 3) containing the (x, y, z, scale)
        at [0, :] contains the center of the cube

        n_cubes = the number of cubes found per dimension

    """

    cubes = []
    centers = [
        np.array([x, y, z], dtype=int)
        for z in range(layout.shape[2])
        for y in range(layout.shape[1])
        for x in range(layout.shape[0])
        if ((x) % 2 == (z) % 2) and ((y) % 2 == (z) % 2)
    ]

    n_cubes = np.zeros((layout.shape[0] // 2))

    for c in centers:
        for cube_vec in constant_cube:
            arr = c + cube_vec
            index = layout.get_node_index(*arr)
            # filter out boundary cases
            if np.any((arr < 0) | np.greater_equal(arr, layout.shape)):
                break
            # filter out nodes that are measured
            if index in removed_nodes:
                break
        else:
            cube = np.empty((19, 3), dtype=int)
            """
            Format:
            cube[0, :] = center of the cube
            cube[:19, :] = coordinates
            """
            cube[0, :] = c
            for i, cube_vec in enumerate(constant_cube):
                cube[i + 1, :3] = c + cube_vec
                # cube[i,  3] = scale
            n_cubes[0] += 1
            cubes.append(cube)
    return cubes


def taxicab_metric(node1, node2):
    x1 = np.array(node1)
    x2 = np.array(node2)

    return np.sum(np.abs(x1 - x2))


def build_centers_graph(cubes, layout):
    """
    Extract the data from the numpy array.

    Returns: the graph of centers C
    """
    C = rx.PyGraph(
        multigraph=False
    )  # C is an object that contains all the linked centers

    for c in cubes:
        C.add_node({"coord": c[0, :]})

    for node_index in C.node_indices():
        for node_index2 in C.node_indices():
            if taxicab_metric(C[node_index]["coord"], C[node_index2]["coord"]) == 2:
                C.add_edge(node_index, node_index2, None)

    return C


def find_max_connected_lattice(C):
    """
    Returns the largest subgraph.
    Input: A connected cube: networkx Graph object that is a graph of centers
    Each node is a tuple (x, y, z)
    """
    try:
        largest_cc = max(rx.connected_components(C), key=len)
    except ValueError:
        largest_cc = rx.PyGraph()
    return largest_cc


def connected_cube_to_nodes(connected_unit_cells_center_graph):
    """
    Expand the graph of centers into full Raussendorf cells.
    """
    connected_all_node_graph = rx.PyGraph()

    for node_index in connected_unit_cells_center_graph.node_indices():
        for cube_vec in constant_cube:
            connected_all_node_graph.add_node(
                {
                    "coord": connected_unit_cells_center_graph[node_index]["coord"]
                    + cube_vec
                }
            )

    for node_index in connected_all_node_graph.node_indices():
        for node_index2 in connected_all_node_graph.node_indices():
            if (
                taxicab_metric(
                    connected_all_node_graph[node_index]["coord"],
                    connected_all_node_graph[node_index2]["coord"],
                )
                == 1
            ):
                connected_all_node_graph.add_edge(node_index, node_index2, None)

    return connected_all_node_graph
