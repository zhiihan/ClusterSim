from clustersim.simulator import ClusterState
import rustworkx as rx
import networkx as nx
import pytest
from tests.test_stabilizers import random_stim_circuit
from hypothesis import given, strategies as st


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
    c.measure(4, basis="Y")
    d = c.to_rustworkx()

    c2 = ClusterState.from_rustworkx(d)

    assert c == c2


def test_load_text():
    text = "H 0 1 2 3 4\nCZ 0 1 1 2 2 3 3 4 4 0\nX 3 4 0\nY 1\nZ 2\nH 3\nS 4\nCZ 4 1\nCX 2 0\n"

    g, log = ClusterState.from_string(text, return_log=True)

    assert log == text
    assert str(g) == "['+ZZZXY', '-XXZXY', '-XZXXI', '-XZZZI', '+XZIIZ']"


def test_load_text_edge():
    text = "H 0 1 2 3 4\nCZ 0 1 1 2 2 3 3 4 4 0\nADD_EDGE 3 4 0 2 1\n"

    g, log = ClusterState.from_string(text, return_log=True)

    assert log == text
    assert str(g) == "['+XZZZZ', '+ZXZZZ', '+ZZXZZ', '+ZZZXZ', '+ZZZZX']"


def test_load_text_add_node():
    text = "ADD_NODE 0\nADD_NODE 1\nADD_NODE 2\n"

    g, log = ClusterState.from_string(text, return_log=True)

    assert log == text
    assert len(g) == 3
    assert str(g) == "['+XII', '+IXI', '+IIX']"


def test_add_node_measure():
    program = """H 0 1 
    CZ 0 1
    ADD_NODE 2
    MX[0] 2
    """
    g1 = ClusterState.from_string(program)
    g2 = ClusterState(2)
    g2.H(0)
    g2.H(1)
    g2.CZ(0, 1)
    g2.add_node()
    g2.measure(2, 0, basis="X")

    assert g1 == g2


def test_duplicate_measure():
    program = """H 0 1 
    CZ 0 1
    DUPLICATE 0 1 
    DUPLICATE 0 1
    MX[0] 5
    """
    g1 = ClusterState.from_string(program)
    g2 = ClusterState(2)
    g2.H(0)
    g2.H(1)
    g2.CZ(0, 1)
    g2 = g2.duplicate([0, 1])
    g2 = g2.duplicate([0, 1])
    g2.measure(5, 0, basis="X")

    assert g1 == g2


@st.composite
def random_graph_program(draw):
    """
    Generate a program string using only ADD_NODE, REMOVE_NODE, and DUPLICATE.
    """
    num_nodes = 0
    instructions = []

    # Generate between 1 and 15 steps
    num_steps = draw(st.integers(min_value=1, max_value=15))

    # Keep track of used node labels for ADD_NODE to ensure they are unique
    used_labels = set()

    for _ in range(num_steps):
        choices = ["ADD_NODE"]
        if num_nodes > 0:
            choices.append("REMOVE_NODE")
            choices.append("DUPLICATE")

        op = draw(st.sampled_from(choices))

        if op == "ADD_NODE":
            k = draw(st.integers(min_value=1, max_value=5))
            labels = []
            for _ in range(k):
                label = draw(
                    st.integers(min_value=0, max_value=1000).filter(
                        lambda x: x not in used_labels
                    )
                )
                used_labels.add(label)
                labels.append(label)
            instructions.append(f"ADD_NODE {' '.join(map(str, labels))}")
            num_nodes += k

        elif op == "REMOVE_NODE":
            active_indices = list(range(num_nodes))
            num_to_remove = draw(st.integers(min_value=1, max_value=num_nodes))
            targets = draw(
                st.lists(
                    st.sampled_from(active_indices),
                    min_size=num_to_remove,
                    max_size=num_to_remove,
                    unique=True,
                )
            )
            instructions.append(f"REMOVE_NODE {' '.join(map(str, targets))}")
            num_nodes -= len(targets)

        elif op == "DUPLICATE":
            active_indices = list(range(num_nodes))
            num_to_dup = draw(st.integers(min_value=1, max_value=num_nodes))
            targets = draw(
                st.lists(
                    st.sampled_from(active_indices),
                    min_size=num_to_dup,
                    max_size=num_to_dup,
                    unique=True,
                )
            )
            instructions.append(f"DUPLICATE {' '.join(map(str, targets))}")
            num_nodes += len(targets)

    return "\n".join(instructions) + "\n"


from hypothesis import settings


@settings(deadline=None)
@given(program=random_graph_program())
def test_load_undo(program):
    """
    Test applying the same sequence of gates as in the stimulus program results in the same graph.
    """
    g = ClusterState.from_string(program)

    # Verify serialization round-trip
    serialized = g.to_json()
    g_reconstructed = ClusterState.from_json(serialized)
    assert g == g_reconstructed
