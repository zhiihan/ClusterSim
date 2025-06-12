import pytest
from cluster_sim.simulator import ClusterState
import networkx as nx
import rustworkx as rx


def test_load_json():
    """
    Test loading a graph from a JSON file.
    """

    g_nx = nx.Graph()
    g_nx.add_nodes_from(range(3))
    g_nx.add_edges_from([(0, 1), (1, 2)])

    G = ClusterState(g_nx)
    json_data = G.to_json()
    G2 = ClusterState.from_json(json_data)
    for node1, node2 in zip(G.graph.node_indices(), G2.graph.node_indices()):
        print(node1, node2, G.graph[node1], G2.graph[node2])
        assert (
            G.graph[node1] == G2.graph[node2]
        ), f"Node {node1} in G and node {node2} in G2 should match."


@pytest.mark.parametrize("circuit,result", [(["h"], 0), (["h", "x"], 1)])
def test_apply_gate(circuit, result):
    """
    Test measurement. Also tests importing from networkx and rustworkx.
    """
    g_nx = nx.Graph()
    g_nx.add_nodes_from(range(5))

    g_rx = rx.PyGraph()
    g_rx.add_nodes_from(range(5))

    G = ClusterState(g_nx)
    H = ClusterState(g_rx)

    for i in range(5):
        for gate in circuit:
            getattr(G, gate)(i)
            getattr(H, gate)(i)

    for i in range(5):
        outcome = G.measure(i, basis="Z")
        outcome2 = H.measure(i, basis="Z")
        assert (
            outcome == outcome2
        ), f"Measurement outcome {outcome} and {outcome2} should match."
        assert outcome == result, f"Measurement outcome {outcome} is not valid."


def test_random_outcome():
    """
    Test measurement.
    """

    samples = 10000

    g_nx = nx.Graph()
    g_nx.add_nodes_from(range(samples))

    G = ClusterState(g_nx)

    outcome = 0
    for i in range(samples):
        outcome += G.measure(i, basis="Z")

    assert outcome == pytest.approx(
        samples * 0.5, rel=0.1
    ), f"Measurement outcome {outcome} is not valid."


def test_cx():
    """
    Test measurement.
    """

    samples = 100

    for i in range(samples):
        g_nx = nx.Graph()
        g_nx.add_nodes_from(range(2))

        G = ClusterState(g_nx)

        G.h(1)
        G.cx(0, 1)

        # Final state should be |00> + |11>

        q1 = G.measure(0, basis="Z")
        q2 = G.measure(1, basis="Z")

        assert q1 == q2, f"Measurement outcome {q1} and {q2} should match."
