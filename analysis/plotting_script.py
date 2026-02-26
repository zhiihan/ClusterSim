import rustworkx as rx
from cluster_sim.app import (
    rx_graph_to_plot,
    grid_graph_3d,
    Grid3D,
    BrowserState,
    update_plot_plotly,
)
import plotly.graph_objects as go
from cluster_sim import ClusterState
from algorithms import find_lattice, connected_cube_to_nodes, build_centers_graph
import random
import numpy as np

b = BrowserState()

b.p_err = 0.01
b.shape = (11, 11, 11)

layout = Grid3D(b)
SEED = 1
random.seed(1)

for i in range(b.shape[0] * b.shape[1] * b.shape[2]):
    if random.random() < b.p_err:
        b.removed_nodes.add(i)

cubes = find_lattice(layout, b.removed_nodes)

C = build_centers_graph(cubes, layout)
connected_cubes = max(
    [C.subgraph(list(c)) for c in rx.connected_components(C)], key=len
)

X = connected_cube_to_nodes(connected_cubes)

D = rx.PyGraph()
for i in b.removed_nodes:
    D.add_node({"coord": layout.get_node_coords(i)})

D_nodes, D_edges, _ = rx_graph_to_plot(D, browser_state=b)
nodes, edges, _ = rx_graph_to_plot(X, browser_state=b)

x_min = np.array([np.inf, np.inf, np.inf])
x_max = np.array([0, 0, 0])
for i in X.node_indices():
    x_min = np.minimum(X[i]["coord"], x_min)
    x_max = np.maximum(X[i]["coord"], x_max)
print(f"Percolated a distance of {x_max - x_min}")

lattice = go.Scatter3d(
    x=nodes[:, 0],
    y=nodes[:, 1],
    z=nodes[:, 2],
    mode="markers",
    line=dict(color="blue", width=2),
    hoverinfo="none",
)

lattice_edges = go.Scatter3d(
    x=edges[:, 0],
    y=edges[:, 1],
    z=edges[:, 2],
    mode="lines",
    line=dict(color="blue", width=2),
    hoverinfo="none",
)

lattice_D = go.Scatter3d(
    x=D_nodes[:, 0],
    y=D_nodes[:, 1],
    z=D_nodes[:, 2],
    mode="markers",
    line=dict(color="green", width=2),
    hoverinfo="none",
)

lattice_edges_D = go.Scatter3d(
    x=D_edges[:, 0],
    y=D_edges[:, 1],
    z=D_edges[:, 2],
    mode="lines",
    line=dict(color="green", width=2),
    hoverinfo="none",
)

fig = update_plot_plotly([lattice, lattice_edges, lattice_D, lattice_edges_D], b)
fig
