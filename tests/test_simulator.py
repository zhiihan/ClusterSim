import pytest
from cluster_sim.simulator import ClusterState
import rustworkx as rx
import networkx as nx

def test_apply_gate():
    """
    Test measurement.
    """

    g = rx.PyGraph()
    g.add_nodes_from([0, 1, 2, 3, 4])

    G = ClusterState(5)

    for i in range(5):
        G.X(i)

    for i in range(5):
        outcome = G.measure(i, basis="Z")
        assert outcome == 1, f"Measurement outcome {outcome} is not valid."


def test_linear_graph():
    g = rx.generators.path_graph(5)
    G = ClusterState.from_rustworkx(g)

    assert G.stabilizers == ["+XZIII", "+ZXZII", "+IZXZI", "+IIZXZ", "+IIIZX"]

def test_import_export_rustworkx():
    g = rx.generators.grid_graph(3, 3)
    c = ClusterState.from_rustworkx(g)
    c.measure(4, basis='Y')
    d = c.to_rustworkx()

    c2 = ClusterState.from_rustworkx(d)

    assert c == c2

def test_import_export_json():

    g = rx.generators.grid_graph(3, 3)
    c = ClusterState.from_rustworkx(g)
    c.measure(4, basis='Y')
    d = c.to_json()

    c2 = ClusterState.from_json(d)

    assert c == c2

def test_import_export_networkx():
    g = nx.grid_2d_graph(3, 2)
    c = ClusterState.from_networkx(g)
    c.measure(4, basis='Y')
    d = c.to_networkx()

    c2 = ClusterState.from_networkx(d)
    assert c == c2

def test_import_export_cytoscape():
    g = nx.grid_2d_graph(3, 2)

    c = ClusterState.from_networkx(g)
    c.H(0)
    c.measure(4, basis='Y')
    d = c.to_cytoscape()
    c2 = ClusterState.from_cytoscape(d)
    d2 = c2.to_cytoscape()
    c3 = ClusterState.from_cytoscape(d2)
    assert c3 == c2