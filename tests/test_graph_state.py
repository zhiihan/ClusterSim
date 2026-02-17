import pytest
from cluster_sim.graph_state import GraphState


def test_measurement():
    """
    Test measurement.
    """

    G = GraphState(5)

    for i in range(5):
        G.x(i)

    for i in range(5):
        outcome = G.measure(i, basis="Z")
        assert outcome == 1, f"Measurement outcome {outcome} is not valid."


def test_linear_graph_state():
    """
    Verify the stabilizers of a graph state are X_i Z_j = 1
    """

    def setup_graph_state():
        G = GraphState(5)

        for i in range(5):
            G.h(i)

        for i in range(4):
            G.cz(i, (i+1))

        return G


    for i in range(5):
        G = setup_graph_state()
        neighbors = tuple(G.vertices[i].neighbors)

        s = 1

        s*= (-1)**G.measure(target=i, basis='X')
        for j in neighbors:
            s*= (-1)**G.measure(j, basis='Z')

        assert s == 1