import rustworkx as rx
from cluster_sim.app import Grid3D, BrowserState
from algorithms import find_lattice, build_centers_graph
import random
import numpy as np
import matplotlib.pyplot as plt

SEED = 1
random.seed(SEED)

# n_target_shape = (3, 3, 3)
n_target_cubes = 27

shapes = []
p_err_array = np.linspace(0, 0.12, 13)

n_samples = 3

size_array = np.arange(0, 30, 2)

for p_err in p_err_array:
    run_results = []
    for run in range(n_samples):
        for size in size_array:
            b = BrowserState()
            b.p_err = p_err
            b.shape = (5 + size, 5 + size, 5 + size)

            layout = Grid3D(b)

            # Generate boolean mask in one vectorized call
            mask = np.random.random(b.shape[0] * b.shape[1] * b.shape[2]) < p_err
            removed_nodes = np.nonzero(mask)[0]

            cubes = find_lattice(layout, removed_nodes)
            C = build_centers_graph(cubes, layout)

            largest_subgraph = [C.subgraph(list(c)) for c in rx.connected_components(C)]
            if not largest_subgraph:
                # print('Not a single connected unit cell found!')
                continue

            F = max(largest_subgraph, key=len)
            x_min = np.array([np.inf, np.inf, np.inf])
            x_max = np.array([0, 0, 0])
            for i in F.node_indices():  # ty:ignore[unresolved-attribute]
                x_min = np.minimum(C[i]["coord"], x_min)
                x_max = np.maximum(C[i]["coord"], x_max)

            # print(f'For {b.shape}, Percolated a distance of {x_max - x_min}')
            # print(f'Found unit_cells {len(F)}, {b.p_err}')

            if len(F) >= n_target_cubes:  # Looking for the largest connected graph
                # print(f'Need {b.shape} to make n_target_cubes >= {n_target_cubes}, made {len(F)}')
                run_results.append(b.shape[0])
                break
        else:
            run_results.append(b.shape[0])
    run_results_avg = np.mean(run_results)
    print(f"Completed, {p_err}, {run_results_avg}")
    shapes.append(run_results_avg)

data = [(p, s) for p, s in zip(p_err_array, shapes)]
np.savetxt("shapes.csv", np.array(data), delimiter=",")
plt.plot(p_err_array, shapes)
plt.title("Largest cubic lattice that generates connected RHG lattice of 27 unit cells")
plt.ylabel("Cubic lattice L (qubits = L^3)")
plt.xlabel("Physical loss p_err")
plt.savefig("figure.png")
