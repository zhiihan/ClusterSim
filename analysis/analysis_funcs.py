import random
import numpy as np
import networkx as nx
from cluster_sim.app.utils import get_node_index, get_node_coords, taxicab_metric
from cluster_sim.app.holes import Holes
import itertools
import logging

from numba import jit


def apply_error_channel(p, seed, shape, removed_nodes, G):
    """
    Randomly measure qubits.
    """
    D = Holes(shape)

    random.seed(int(seed))
    # p is the probability of losing a qubit

    for i in range(shape[0] * shape[1] * shape[2]):
        if random.random() < p:
            if removed_nodes[i] == False:  # noqa: E712
                removed_nodes[i] = True
                D.add_node(i)
                G.handle_measurements(i, "Z")
                G.graph.remove_node(get_node_coords(i, shape))

    return G, D, removed_nodes


def rhg_lattice_scale(G, D, removed_nodes, shape, scale_factor=1):
    """
    Create a RHG lattice from a square lattice, with a scale factor.

    Parameters:
    - nclicks: number of clicks on the button (unused)
    - scale_factor: scale factor for the RHG lattice
    - browser_data: data from the browser
    - graphData: data from the graph
    - holeData: data from the holes
    """
    holes = D.graph.nodes

    hole_locations = np.zeros(
        (scale_factor + 1, scale_factor + 1, scale_factor + 1), dtype=int
    )

    # Finding the offset that maximizes holes placed in hole locations
    for h in holes:
        x, y, z = h

        x_vec = np.array([x, y, z])
        for xoffset, yoffset, zoffset in itertools.product(
            range(scale_factor + 1), repeat=3
        ):
            test_cond = x_vec % (scale_factor + 1)
            offset = np.array([xoffset, yoffset, zoffset])

            if np.all(test_cond == offset) or np.all(test_cond != offset):
                hole_locations[tuple(offset)] += 1

    # print("hole locations", hole_locations)

    # Finding the offset that maximizes holes placed in hole locations
    # Can use other indices to find other maximizing offsets
    offset = np.argwhere(hole_locations == np.max(hole_locations))[0]

    # Measuring the qubits based on the offset
    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                x_vec = np.array([x, y, z]) % (scale_factor + 1)

                offset = np.array(offset)

                if np.all(x_vec == offset) or np.all(x_vec != offset):
                    i = get_node_index(x, y, z, shape)
                    if removed_nodes[i] == False:  # noqa: E712
                        G.handle_measurements(i, "Z")
                        removed_nodes[i] = True

    return G, D, removed_nodes, offset


def reduce_lattice(G, shape, offsets, removed_nodes, scale_factor=1):
    valid_unit_cells, _ = generate_unit_cell_global_coords(shape, scale_factor, offsets)

    C = nx.Graph()
    graphs_hashmap = {}
    imperfect_cells = 0
    all_graphs = []
    perfect_cells = 0

    for unit_cell_coord in valid_unit_cells:
        H_subgraph, imperfection_score = find_rings(
            G, scale_factor, removed_nodes, shape, unit_cell_coord=unit_cell_coord
        )
        all_graphs.append(H_subgraph)
        if imperfection_score == 0:
            C.add_node(tuple(unit_cell_coord))
            graphs_hashmap[tuple(unit_cell_coord)] = H_subgraph
            perfect_cells += 1

            adjacent_nodes_list = adjacent_nodes(
                unit_cell_coord=unit_cell_coord, scale_factor=scale_factor
            ).tolist()

            for c in adjacent_nodes_list:
                if (
                    taxicab_metric(np.array(c), np.array(unit_cell_coord))
                    <= (scale_factor + 1)
                    and tuple(c) in C.nodes
                ):
                    C.add_edge(tuple(c), tuple(unit_cell_coord))
        else:
            imperfect_cells += 1

    largest_cc = max(nx.connected_components(C), key=len)
    largest_cc = C.subgraph(largest_cc).copy()

    # Check if the largest cluster percolates
    low = np.array([np.inf, np.inf, np.inf])
    high = np.zeros(3)

    if not largest_cc:
        # print("No clusters")
        percolation_distance = 0

    else:
        for node in largest_cc.nodes:
            # Get the coordinates of the node
            low = np.minimum(low, np.array(node))
            high = np.maximum(high, np.array(node))
        percolation_distance = high[2] - low[2]

        connected_perfect_cells = largest_cc.number_of_nodes()

    H_all = nx.compose_all(all_graphs)

    return (
        H_all,
        perfect_cells,
        connected_perfect_cells,
        imperfect_cells,
        percolation_distance,
    )


def generate_ring(scale_factor: int, global_offset: np.ndarray, j: int, ring_dir: str):
    """
    Generate the rings of a unit cell in a Raussendorf lattice.
    """

    ring_coords = np.empty((4 * (scale_factor + 2), 3), dtype=int)

    for i in range(0, scale_factor + 2):
        ring_coords[4 * i : 4 * (i + 1)] = (  # noqa: E203
            ring_gen_funcs(i, j, scale_factor, ring_dir) + global_offset
        )

    ring_coords = np.unique(ring_coords, axis=0)
    ring_coords = [tuple(coord) for coord in ring_coords]

    return ring_coords


@jit(nopython=True)
def ring_gen_funcs(i: int, j: int, scale_factor: int, ring_dir: str) -> np.ndarray:
    """
    Generate the coordinates of a ring.
    """

    if ring_dir == "x":
        return np.array(
            [[j, 0, i], [j, i, scale_factor + 1], [j, scale_factor + 1, i], [j, i, 0]],
        )
    elif ring_dir == "y":
        return np.array(
            [[0, j, i], [i, j, scale_factor + 1], [scale_factor + 1, j, i], [i, j, 0]],
        )
    elif ring_dir == "z":
        return np.array(
            [[0, i, j], [i, scale_factor + 1, j], [scale_factor + 1, i, j], [i, 0, j]],
        )


@jit(nopython=True)
def adjacent_nodes(unit_cell_coord: np.ndarray, scale_factor: int) -> np.ndarray:

    x0 = unit_cell_coord + np.array(
        [
            [scale_factor + 1, 0, 0],
            [0, scale_factor + 1, 0],
            [0, 0, scale_factor + 1],
            [-(scale_factor + 1), 0, 0],
            [0, -(scale_factor + 1), 0],
            [0, 0, -(scale_factor + 1)],
        ]
    )

    return x0


def find_rings(G, scale_factor: int, removed_nodes, shape, unit_cell_coord=(0, 0, 0)):
    """
    Find the rings of a unit cell in a Raussendorf lattice.
    """

    unit_cell_coord = np.array(unit_cell_coord)

    optimized_rings = []
    imperfection_score = 0

    for ring_gen in ["x", "y", "z"]:
        rings = {}
        for j in range(1, scale_factor + 1):
            ring_list = generate_ring(scale_factor, unit_cell_coord, j, ring_gen)
            counter = evaluate_ring(ring_list, removed_nodes, shape)

            if counter == 0:
                optimized_rings.append(ring_list)
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

    logging.info(f"Total Imperfection Score: {imperfection_score}")

    return G.graph.subgraph(optimized_rings), imperfection_score


def evaluate_ring(ring_list, removed_nodes, shape) -> int:
    """
    Evaluate the number of nodes in a ring that are not in the graph.
    This is used to determine how many nodes are missing from the graph.
    """
    counter = 0  # noqa: E712
    for node in ring_list:
        node = get_node_index(node[0], node[1], node[2], shape)
        if removed_nodes[node]:
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
        print(f"Unit cell location: {i}, scale factor: {scale_factor}")

        if any(
            np.array(i) * (scale_factor + 1) + (scale_factor + 2) + offset
            > np.array(shape)
        ):
            print(f"Skipping unit cell {i} as it exceeds grid dimensions.")
            continue

        unit_cell_locations.append(np.array(i) * (scale_factor + 1) + offset)

        unit_cell_shape = np.maximum(np.array(i), unit_cell_shape)

    return unit_cell_locations, unit_cell_shape + np.array([1, 1, 1])
