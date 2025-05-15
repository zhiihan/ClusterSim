import pytest
from cluster_sim.simulator import ClusterState
import networkx as nx


@pytest.mark.parametrize("circuit,result", [(["h"], 0), (["h", "x"], 1)])
def test_apply_gate(circuit, result):
    """
    Test measurement.
    """
    g_nx = nx.Graph()
    g_nx.add_nodes_from(range(5))

    G = ClusterState(g_nx)

    for i in range(5):
        for gate in circuit:
            getattr(G, gate)(i)

    for i in range(5):
        outcome = G.measure(i, basis="Z")
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
