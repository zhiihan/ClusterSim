import random
import numpy as np
import networkx as nx
from cluster_sim.app.utils import get_node_index, get_node_coords
from cluster_sim.app.holes import Holes
from cluster_sim.app.grid import Grid
import tqdm
from joblib import Parallel
import itertools


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
                G.graph.remove_node(get_node_coords(i, shape))

    return G, D, removed_nodes


class ParallelTqdm(Parallel):
    """joblib.Parallel, but with a tqdm progressbar

    Additional parameters:
    ----------------------
    total_tasks: int, default: None
        the number of expected jobs. Used in the tqdm progressbar.
        If None, try to infer from the length of the called iterator, and
        fallback to use the number of remaining items as soon as we finish
        dispatching.
        Note: use a list instead of an iterator if you want the total_tasks
        to be inferred from its length.

    desc: str, default: None
        the description used in the tqdm progressbar.

    disable_progressbar: bool, default: False
        If True, a tqdm progressbar is not used.

    show_joblib_header: bool, default: False
        If True, show joblib header before the progressbar.

    Removed parameters:
    -------------------
    verbose: will be ignored


    Usage:
    ------
    >>> from joblib import delayed
    >>> from time import sleep
    >>> ParallelTqdm(n_jobs=-1)([delayed(sleep)(.1) for _ in range(10)])
    80%|████████  | 8/10 [00:02<00:00,  3.12tasks/s]

    """

    def __init__(
        self,
        *,
        total_tasks: int | None = None,
        desc: str | None = None,
        disable_progressbar: bool = False,
        show_joblib_header: bool = False,
        **kwargs,
    ):
        if "verbose" in kwargs:
            raise ValueError(
                "verbose is not supported. "
                "Use disable_progressbar and show_joblib_header instead."
            )
        super().__init__(verbose=(1 if show_joblib_header else 0), **kwargs)
        self.total_tasks = total_tasks
        self.desc = desc
        self.disable_progressbar = disable_progressbar
        self.progress_bar: tqdm.tqdm | None = None

    def __call__(self, iterable):
        try:
            if self.total_tasks is None:
                # try to infer total_tasks from the length of the called iterator
                try:
                    self.total_tasks = len(iterable)
                except (TypeError, AttributeError):
                    pass
            # call parent function
            return super().__call__(iterable)
        finally:
            # close tqdm progress bar
            if self.progress_bar is not None:
                self.progress_bar.close()

    __call__.__doc__ = Parallel.__call__.__doc__

    def dispatch_one_batch(self, iterator):
        # start progress_bar, if not started yet.
        if self.progress_bar is None:
            self.progress_bar = tqdm.tqdm(
                desc=self.desc,
                total=self.total_tasks,
                disable=self.disable_progressbar,
                unit="tasks",
            )
        # call parent function
        return super().dispatch_one_batch(iterator)

    dispatch_one_batch.__doc__ = Parallel.dispatch_one_batch.__doc__

    def print_progress(self):
        """Display the process of the parallel execution using tqdm"""
        # if we finish dispatching, find total_tasks from the number of remaining items
        if self.total_tasks is None and self._original_iterator is None:
            self.total_tasks = self.n_dispatched_tasks
            self.progress_bar.total = self.total_tasks
            self.progress_bar.refresh()
        # update progressbar
        self.progress_bar.update(self.n_completed_tasks - self.progress_bar.n)


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
                    if removed_nodes[i] == False:
                        G.handle_measurements(i, "Z")
                        removed_nodes[i] = True

    return G, D, removed_nodes, offset


def find_unit_cell(G, shape, offset, scale_factor=1):
    """
    Find unit cells in a 3D grid.

    Returns the number of valid unit cells.
    """

    possible_unit_cells = generate_unit_cell_global_coords(shape, scale_factor)
    valid_unit_cell_counts = 0

    invalid_unit_cell_counts = 0

    for possible_unit in possible_unit_cells:

        unitcell = check_unit_cell(
            G, scale_factor, offset, unit_cell_coord=possible_unit
        )
        if unitcell:
            valid_unit_cell_counts += 1
    return valid_unit_cell_counts


def generate_unit_cell_faces(scale_factor, offset, unit_cell_coord=(0, 0, 0)):
    """
    Generate face slices for a unit cell.
    """
    unit_cell_coord = np.array(unit_cell_coord)

    global_coordinate_offset = np.array(offset) + np.array(unit_cell_coord)

    face_gen_func = [
        lambda d, j: (j, 0, d),  # face1
        lambda d, j: (j, d, 0),  # face2
        lambda d, j: (j, scale_factor + 1, d),  # face3
        lambda d, j: (j, d, scale_factor + 1),  # face4
        lambda d, j: (d, j, 0),  # face5
        lambda d, j: (0, j, d),  # face6
        lambda d, j: (d, j, scale_factor + 1),  # face7
        lambda d, j: (scale_factor + 1, j, d),  # face8
        lambda d, j: (d, scale_factor + 1, j),  # face9
        lambda d, j: (scale_factor + 1, d, j),  # face10
        lambda d, j: (d, 0, j),  # face11
        lambda d, j: (0, d, j),  # face12
    ]

    all_faces = []

    for f in range(12):
        face = []
        for d in range(1, scale_factor + 1):
            face.append(
                [
                    tuple(global_coordinate_offset + np.array(face_gen_func[f](d, j)))
                    for j in range(0, scale_factor + 2)
                ]
            )
        all_faces.append(face)

    all_faces_unzipped = [
        node for face in all_faces for checks in face for node in checks
    ]
    return all_faces, all_faces_unzipped, face_gen_func, global_coordinate_offset


def check_unit_cell(G, scale_factor, offset, unit_cell_coord=(0, 0, 0)):
    """
    Check if a unit cell is a valid Raussendorf unit cell.
    A valid unit cell is a cube has 6 faces, each with 2 orientations, for a total of 12 oriented faces.

    If all oriented faces contains at least 1 line that does not contain an erasure, the unit cell is valid.
    """

    all_faces, all_faces_unzipped, _, _ = generate_unit_cell_faces(
        scale_factor, offset, unit_cell_coord=unit_cell_coord
    )

    H = G.graph.subgraph(all_faces_unzipped).copy()
    H.remove_nodes_from(list(nx.isolates(H)))

    joined_faces = []

    for face in all_faces:
        for checks in face:
            if all(H.has_node(i) for i in checks):
                joined_faces.append(checks)
                break
        else:
            # print("No face found")
            return None

    joined_faces = [node for l in joined_faces for node in l]

    return G.graph.subgraph(joined_faces)


def generate_unit_cell_global_coords(shape, scale_factor):
    """
    Find the bottom left corner of each unit cell in a 3D grid.
    """

    # Calculate the number of cubes in each dimension
    num_cubes = np.array(shape) // (scale_factor + 1)

    unit_cell_locations = []
    for i in itertools.product(
        range(num_cubes[0]), range(num_cubes[1]), range(num_cubes[2])
    ):
        unit_cell_locations.append(np.array(i) * (scale_factor + 1))

    return unit_cell_locations


def check_unit_cell_path(G, scale_factor, offset, unit_cell_coord=(0, 0, 0)):
    """
    Check if a unit cell is a valid Raussendorf unit cell.
    A valid unit cell is a cube has 6 faces, each with 2 orientations, for a total of 12 oriented faces.

    If all oriented faces contains at least 1 line that does not contain an erasure, the unit cell is valid.

    This is the path version of the function, which may run slower.
    """

    all_faces, all_faces_unzipped, face_gen_func, global_coordinate_offset = (
        generate_unit_cell_faces(scale_factor, offset, unit_cell_coord=unit_cell_coord)
    )

    H = G.graph.subgraph(all_faces_unzipped).copy()
    H.remove_nodes_from(list(nx.isolates(H)))

    joined_faces = []

    for face, gen_func in zip(all_faces, face_gen_func):
        face_unzipped = [node for checks in face for node in checks]

        F = H.subgraph(face_unzipped).copy()

        edges1 = [
            tuple(global_coordinate_offset + np.array(gen_func(d, 0)))
            for d in range(1, scale_factor + 1)
        ]
        edges2 = [
            tuple(global_coordinate_offset + np.array(gen_func(d, scale_factor + 1)))
            for d in range(1, scale_factor + 1)
        ]

        for i, j in itertools.product(edges1, edges2):

            # Edge cases for when the unit cell is at the edge of the grid
            if i not in F.nodes:
                continue
            if j not in F.nodes:
                continue

            if nx.has_path(F, i, j):
                path = nx.shortest_path(F, i, j)
                joined_faces.append(path)
                # should append path
                break
        else:
            # print("No face found")
            return None

    joined_faces = [node for l in joined_faces for node in l]

    return G.graph.subgraph(joined_faces)
