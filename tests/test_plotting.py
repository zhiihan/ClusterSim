from cluster_sim.simulator import ClusterState
import networkx as nx

from cluster_sim.plotting import Plot3DGrid


def test_plotting():
    """
    Test plotting a 3D grid graph using Plot3DGrid.
    """
    g = nx.grid_2d_graph(3, 3)

    g = ClusterState(g)

    Plot3DGrid(g, shape=(3, 3, 1)).plot()
