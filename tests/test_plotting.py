import numpy as np
from cluster_sim.app.layout import Grid3D
import rustworkx as rx
from cluster_sim.simulator import ClusterState

def test_plotting():
    nodes = np.array([[0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [1., 1., 0.]])

    edges =     np.array([[ 0.,  0.,  0.],
            [ 1.,  0.,  0.],
            [np.nan, np.nan, np.nan],
            [ 0.,  0.,  0.],
            [ 0.,  1.,  0.],
            [np.nan, np.nan, np.nan],
            [ 1.,  0.,  0.],
            [ 1.,  1.,  0.],
            [np.nan, np.nan, np.nan],
            [ 0.,  1.,  0.],
            [ 1.,  1.,  0.],
            [np.nan, np.nan, np.nan]])

    g = rx.generators.grid_graph(2, 2)
    g = ClusterState.from_rustworkx(g).to_rustworkx()
    g_nodes, g_edges = Grid3D(g, shape=(2, 2, 1)).graph_to_plot()
    assert np.allclose(g_nodes, nodes, equal_nan=True)
    assert np.allclose(g_edges, edges, equal_nan=True)