import rustworkx as rx


def grid_graph_3d(shape: tuple[int, int, int]):
    """Generate a Grid graph in rustworkx

    Args:
        shape (tuple[int, int, int]): the shape

    Returns:
        _type_: rustworkx graph
    """
    Lx, Ly, Lz = shape
    G = rx.PyGraph()
    G.add_nodes_from(
        [(i, j, k) for k in range(Lz) for j in range(Ly) for i in range(Lx)]
    )
    G.add_edges_from(
        [
            (i + j * Lx + k * Lx * Ly, (i + 1) + j * Lx + k * Lx * Ly, None)
            for k in range(Lz)
            for j in range(Ly)
            for i in range(Lx - 1)
        ]
    )
    G.add_edges_from(
        [
            (i + j * Lx + k * Lx * Ly, i + (j + 1) * Lx + k * Lx * Ly, None)
            for k in range(Lz)
            for j in range(Ly - 1)
            for i in range(Lx)
        ]
    )
    G.add_edges_from(
        [
            (i + j * Lx + k * Lx * Ly, i + j * Lx + (k + 1) * Lx * Ly, None)
            for k in range(Lz - 1)
            for j in range(Ly)
            for i in range(Lx)
        ]
    )
    return G
