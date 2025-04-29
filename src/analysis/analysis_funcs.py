import random
import numpy as np
import networkx as nx
from cluster_sim.app.utils import get_node_index
from cluster_sim.app.holes import Holes
from cluster_sim.app.grid import Grid
import tqdm
from joblib import Parallel


def path_percolation(G, removed_nodes, shape, xoffset, yoffset):
    """
    Check if there is a path from the bottom to the top of the lattice.

    Returns:
        percolates: 1 if there is a path, 0 otherwise
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
            percolates = 1
            break
    else:
        percolates = 0

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
