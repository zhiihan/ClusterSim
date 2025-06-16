import pytest
from cluster_sim.graph_state import GraphState


def test_apply_gate():
    """
    Test measurement.
    """

    G = GraphState(5)

    for i in range(5):
        G.x(i)

    for i in range(5):
        outcome = G.measure(i, basis="Z")
        assert outcome == 1, f"Measurement outcome {outcome} is not valid."
