import random
import numpy as np
import networkx as nx
from cluster_sim.app.utils import get_node_index
from cluster_sim.app.holes import Holes
from cluster_sim.app.grid import Grid


def path_percolation(G, removed_nodes, shape, xoffset, yoffset):
    """
    Check if there is a path from the bottom to the top of the lattice.
    """
    gnx = G.to_networkx()

    removed_nodes_reshape = removed_nodes.reshape(shape)

    zmax = shape[2]

    zeroplane = removed_nodes_reshape[:, :, 0]
    zmaxplane = removed_nodes_reshape[:, :, zmax - 1]

    start = np.argwhere(
        zeroplane == 0
    )  # This is the coordinates of all valid node in z = 0
    end = np.argwhere(
        zmaxplane == 0
    )  # This is the coordinates of all valid node in z = L

    for index in range(len(end)):
        i = get_node_index(0 + xoffset, 1 + yoffset, 0, shape)
        j = get_node_index(*end[index], zmax - 1, shape)
        if nx.has_path(gnx, i, j):
            percolates = True
            break
    else:
        percolates = False

    return percolates


def apply_error_channel(p, seed, shape, removed_nodes, G):
    """
    Randomly measure qubits.
    """
    D = Holes(shape)

    random.seed(int(seed))
    # p is the probability of losing a qubit

    for i in range(shape[0] * shape[1] * shape[2]):
        if random.random() < p:
            if removed_nodes[i] == False:
                removed_nodes[i] = True
                D.add_node(i)
                G.handle_measurements(i, "Z")

    return G, D, removed_nodes


def algorithm1(G, D, removed_nodes, shape):
    holes = D.graph.nodes
    hole_locations = np.zeros(8)

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

    xoffset = int(np.argmax(hole_locations) % 2)
    yoffset = int(np.argmax(hole_locations) // 2)
    zoffset = int(np.argmax(hole_locations) // 4)

    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                if ((x + xoffset) % 2 == (z + zoffset) % 2) and (
                    (y + yoffset) % 2 == (z + zoffset) % 2
                ):
                    i = get_node_index(x, y, z, shape)
                    removed_nodes[i] = True
                    G.handle_measurements(i, "Z")

    return G, removed_nodes, [xoffset, yoffset, zoffset]
