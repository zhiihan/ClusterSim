import numpy as np
import os
import time
from analysis_funcs import (
    rhg_lattice_scale,
    apply_error_channel,
    reduce_lattice,
)


import networkx as nx
from cluster_sim.app.grid import Grid
import pandas as pd
import sys

os.makedirs("./data", exist_ok=True)

SHAPE = [40, 40, 100]
SEED = 1
NUM_SAMPLES = 1
MAX_SCALE = 5


# Input vector for all jobs
jobs_input_vec = [
    (p, scale, sample)
    for p in np.linspace(0, 0.3, 10)
    for scale in range(1, MAX_SCALE + 1)
    for sample in range(NUM_SAMPLES)
]


def main(input_params):
    """
    Main function for parallel processing. Here, we:

    1. Create a grid of the given SHAPE
    2. Apply the error channel to the grid
    3. Generate a RHG lattice
    4. Look at clusters of the RHG lattice
    5. Return the results

    Returns:
        - p: The input parameter for the simulation
        - percolates: The number of times percolation occurred (True: 1, False: 0)
        - times: The average time taken for the simulation
    """

    p, scale, sample = input_params

    # Sanity check: check that this is equal to the move_list on the app
    # print(np.reSHAPE(np.argwhere(removed_nodes == True), SHAPE=-1))

    # store the outputs for 1 simulation
    data_out = []

    start = time.time()

    G = Grid(SHAPE)
    removed_nodes = np.zeros(SHAPE[0] * SHAPE[1] * SHAPE[2], dtype=bool)

    G, D, removed_nodes = apply_error_channel(p, SEED + sample, SHAPE, removed_nodes, G)
    # Generate an RHG lattice out of G
    removed_nodes_from_erasure_errors = np.sum(removed_nodes)

    setup_time = time.time() - start
    start = time.time()

    G, D, removed_nodes, offsets = rhg_lattice_scale(
        G, D, removed_nodes, SHAPE, scale_factor=scale
    )

    removed_nodes_from_setting_lattice = (
        np.sum(removed_nodes) - removed_nodes_from_erasure_errors
    )
    algo_time = time.time() - start
    start = time.time()

    (
        H_all,
        perfect_cells,
        connected_perfect_cells,
        imperfect_cells,
        percolation_distance,
    ) = reduce_lattice(G, SHAPE, offsets, removed_nodes, scale_factor=scale)

    search_time = time.time() - start

    data_out.append(
        {
            "sample": sample,
            "p": p,
            "algo_time": algo_time,
            "search_time": search_time,
            "setup_time": setup_time,
            "connected_perfect_d_cells": connected_perfect_cells,
            "perfect_d_cells": perfect_cells,
            "imperfect_d_cells": imperfect_cells,
            "imperfection_score": nx.number_of_isolates(H_all),
            "number_of_nodes_final_lattice": H_all.number_of_nodes(),
            "removed_nodes_from_erasure_errors": removed_nodes_from_erasure_errors,
            "removed_nodes_from_setting_lattice": removed_nodes_from_setting_lattice,
            "percolation_distance": percolation_distance,
            "unit_cube_scale": scale,
            "SHAPE_x": SHAPE[0],
            "SHAPE_y": SHAPE[1],
            "SHAPE_z": SHAPE[2],
            "SEED": SEED + sample,
        }
    )

    print(
        f"p = {p}, percolation_distance = {percolation_distance}, algo_time = {algo_time}, search_time = {search_time}, setup_time = {setup_time}, sample = {sample}, unit_cells = {perfect_cells}, unit_cube_scale = {scale}, SHAPE = {SHAPE}, SEED = {SEED + sample}"
    )

    return data_out


if __name__ == "__main__":
    i = int(sys.argv[1])  # get the value of the $SLURM_ARRAY_TASK_ID
    results = main(jobs_input_vec[i])

    name = time.strftime("%Y_%m_%d", time.gmtime())

    exists = os.path.exists(f"./data/{name}.csv")

    df = pd.DataFrame(results)
    df.to_csv(f"./data/{name}.csv", index=False, header=not exists, mode="a")
