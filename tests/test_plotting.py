import numpy as np
from cluster_sim.app import Grid3D, BrowserState, rx_graph_to_plot
import rustworkx as rx
from cluster_sim.simulator import ClusterState


def test_plotting():
    nodes = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )

    edges = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [np.nan, np.nan, np.nan],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [np.nan, np.nan, np.nan],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [np.nan, np.nan, np.nan],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [np.nan, np.nan, np.nan],
        ]
    )

    b = BrowserState()
    b.shape = (2, 2, 1)

    g = rx.generators.grid_graph(2, 2)
    g = ClusterState.from_rustworkx(g).to_rustworkx()

    layout = Grid3D(browser_state=b)

    g = layout.update_graph_with_layout(g)

    g_nodes, g_edges, _ = rx_graph_to_plot(g, browser_state=b)
    assert np.allclose(g_nodes, nodes, equal_nan=True)
    assert np.allclose(g_edges, edges, equal_nan=True)
