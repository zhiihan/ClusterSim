import numpy as np
import os
from joblib import delayed, Parallel
import time
from analysis_funcs import (
    rhg_lattice_scale,
    apply_error_channel,
    ParallelTqdm,
    find_connected_unit_cells,
)
import networkx as nx
import matplotlib.pyplot as plt
from cluster_sim.app.grid import Grid
import pandas as pd
from memory_profiler import profile, memory_usage


os.makedirs("./data", exist_ok=True)

shape = [50, 50, 50]
seed = 1

samples = 1

# Input vector for all jobs
jobs_input_vec = [(p, scale) for scale in range(4, 6) for p in np.linspace(0, 0.3, 30)]


@profile
def main(input_params):
    """
    Main function for parallel processing. Here, we:

    1. Create a grid of the given shape
    2. Apply the error channel to the grid
    3. Generate a RHG lattice
    4. Look at clusters of the RHG lattice
    5. Return the results

    Returns:
        - p: The input parameter for the simulation
        - percolates: The number of times percolation occurred (True: 1, False: 0)
        - times: The average time taken for the simulation
    """

    p, scale = input_params

    # Sanity check: check that this is equal to the move_list on the app
    # print(np.reshape(np.argwhere(removed_nodes == True), shape=-1))

    num_percolates = 0

    # store the outputs for 1 simulation
    data_out = []

    diff = 0

    for i in range(samples):
        percolate = 0
        start = time.time()
        G = Grid(shape)
        removed_nodes = np.zeros(shape[0] * shape[1] * shape[2], dtype=bool)

        G, D, removed_nodes = apply_error_channel(p, seed + i, shape, removed_nodes, G)
        # Generate an RHG lattice out of G
        G, D, removed_nodes, offsets = rhg_lattice_scale(
            G, D, removed_nodes, shape, scale_factor=scale
        )

        C = find_connected_unit_cells(G, shape, offsets, scale_factor=scale)

        if not C:
            data_out.append(
                {
                    "sample": i,
                    "p": p,
                    "times": time.time() - start,
                    "unit_cells": 0,
                    "scale": scale,
                    "percolates": 0,
                    "diff": 0,
                }
            )
            continue

        largest_cc = max(nx.connected_components(C), key=len)
        largest_cc = C.subgraph(largest_cc).copy()

        # Check if the largest cluster percolates
        low = np.array([np.inf, np.inf, np.inf])
        high = np.zeros(3)

        if not largest_cc:
            # print("No clusters")
            diff = 0

        else:
            for node in largest_cc.nodes:
                # Get the coordinates of the node
                low = np.minimum(low, np.array(node))
                high = np.maximum(high, np.array(node))
            diff = high[2] - low[2]

            # print(f"high = {high}, low={low}, diff={diff}")
            if shape[2] - diff <= 2 * (scale + 2):
                num_percolates += 1
                percolate = 1

        end = time.time()

        data_out.append(
            {
                "sample": i,
                "p": p,
                "times": end - start,
                "unit_cells": largest_cc.number_of_nodes(),
                "scale": scale,
                "percolates": percolate,
                "diff": diff,
            }
        )

    print(f"p = {p}, percolates = {num_percolates}, time = {time.time() - start}")

    return data_out


# results = Parallel(n_jobs=-1)([delayed(main)(x) for x in jobs_input_vec])
# results = [item for sublist in results for item in sublist]

# df = pd.DataFrame(results)
# df.to_csv("./data/test.csv", index=False, header=False)


@profile
def grid(shape):
    """
    Create a grid of the given shape
    """
    G = Grid(shape, relabel=False)
    return G


grid(shape)
