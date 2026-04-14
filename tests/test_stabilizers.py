import stim
from cluster_sim.simulator import ClusterState
from hypothesis import given, strategies as st, assume
import pytest


@st.composite
def random_stim_circuit(draw):
    circuit = stim.Circuit()

    # Force at least 2 qubits, up to 10
    num_qubits = draw(st.integers(min_value=2, max_value=10))

    # Number of operations to add
    num_ops = draw(st.integers(min_value=5, max_value=20))

    # Pool of gates: Single-qubit and Two-qubit
    single_qubit_gates = ["H", "X", "Y", "Z", "S", "S_DAG"]
    two_qubit_gates = ["CX", "CZ"]

    for _ in range(num_ops):
        # Randomly choose between a 1-qubit or 2-qubit gate
        gate_type = draw(st.sampled_from(["single", "double"]))

        if gate_type == "single":
            gate = draw(st.sampled_from(single_qubit_gates))
            target = draw(st.integers(min_value=0, max_value=num_qubits - 1))
            circuit.append(gate, [target])

        else:
            gate = draw(st.sampled_from(two_qubit_gates))
            # Pick first qubit
            q1 = draw(st.integers(min_value=0, max_value=num_qubits - 1))
            # Pick second qubit, ensuring it is NOT the same as q1
            q2 = draw(
                st.integers(min_value=0, max_value=num_qubits - 1).filter(
                    lambda x: x != q1
                )
            )
            circuit.append(gate, [q1, q2])

    return circuit


@given(stim_program=random_stim_circuit())
def test_compare_stabilizers_stim(stim_program: stim.Circuit):
    """Compare the canonical stabilizers vs. stim"""
    program = str(stim_program)

    # Stim
    s = stim.TableauSimulator()
    s.do_circuit(stim_program)

    # ClusterSim
    c = ClusterState.from_text(program)
    t = stim.Tableau.from_stabilizers([stim.PauliString(i) for i in c.stabilizers])
    s2 = stim.TableauSimulator()
    s2.do_tableau(t, [i for i in range(len(c))])

    assert s.canonical_stabilizers() == s2.canonical_stabilizers()


@given(data=st.data())
def test_force_measurement(data):
    """Compare the canonical stabilizers vs. stim"""

    stim_program = data.draw(random_stim_circuit())
    program = str(stim_program)
    max_qubit_index = stim_program.num_qubits - 1
    measurement_targets = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=max_qubit_index), min_size=1, unique=True
        )
    )
    force = data.draw(st.sampled_from([0, 1]))

    # Stim
    s = stim.TableauSimulator()
    s.do_circuit(stim_program)
    try:
        s.postselect_z(measurement_targets, desired_value=force)
    except ValueError:
        assume(False)

    # ClusterSim
    c = ClusterState.from_text(program)
    for i in measurement_targets:
        c.M(i, force=force)

    t = stim.Tableau.from_stabilizers([stim.PauliString(i) for i in c.stabilizers])
    s2 = stim.TableauSimulator()
    s2.do_tableau(t, [i for i in range(len(c))])

    assert s.canonical_stabilizers() == s2.canonical_stabilizers()
