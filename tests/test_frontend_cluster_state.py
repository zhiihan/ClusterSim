import pytest
from clustersim.simulator import ClusterState, FrontendClusterState


def test_init_and_gate_delegation():
    state = FrontendClusterState(3)
    assert len(state) == 3
    assert state.num_nodes == 3

    state.H(0)
    state.H(1)
    state.CZ(0, 1)

    assert state.stabilizers == ["+XZI", "+ZXI", "+IIZ"]

    state.MX(2, force=1)
    assert len(state.stabilizers) == 3


def test_node_removal_metadata_transfer():
    state = FrontendClusterState(3)

    pos0 = {"x": 10.0, "y": 20.0}
    pos1 = {"x": 30.0, "y": 40.0}
    pos2 = {"x": 50.0, "y": 60.0}

    state.set_position(0, pos0)
    state.set_position(1, pos1)
    state.set_position(2, pos2)

    state.set_group(2, "group_alpha")
    state.set_value(2, "custom_node_2")

    # Remove node 1
    state.remove_node(1)

    assert len(state) == 2

    # Node 0 retains pos0
    assert state.get_position(0) == pos0

    # Node 1 (former node 2) MUST retain pos2, group_alpha, and custom_node_2
    assert state.get_position(1) == pos2
    assert state.get_group(1) == "group_alpha"
    assert state.get_value(1) == "custom_node_2"


def test_add_node_with_metadata():
    state = FrontendClusterState(2)

    state.add_group("group_beta", label="Beta Group")
    new_idx = state.add_node(
        vop="IA",
        position={"x": 100.0, "y": 200.0},
        group="group_beta",
        value="node_2_val",
    )

    assert new_idx == 2
    assert len(state) == 3
    assert state.get_position(2) == {"x": 100.0, "y": 200.0}
    assert state.get_group(2) == "group_beta"
    assert state.get_value(2) == "node_2_val"


def test_undo_redo_snapshots():
    state = FrontendClusterState(2)
    state.set_position(0, {"x": 0.0, "y": 0.0})
    state.set_position(1, {"x": 10.0, "y": 10.0})

    # Save checkpoint 1
    state.push_state()

    # Perform action: add node 2
    state.add_node(position={"x": 50.0, "y": 50.0})
    assert len(state) == 3

    # Save checkpoint 2
    state.push_state()

    # Perform action: remove node 0
    state.remove_node(0)
    assert len(state) == 2

    # Undo removal -> back to 3 nodes
    assert state.undo() is True
    assert len(state) == 3
    assert state.get_position(2) == {"x": 50.0, "y": 50.0}

    # Undo add node -> back to 2 nodes
    assert state.undo() is True
    assert len(state) == 2

    # Redo -> back to 3 nodes
    assert state.redo() is True
    assert len(state) == 3


def test_cytoscape_elements_roundtrip():
    state = FrontendClusterState(3)
    state.add_group("group_main", label="Main Container")
    state.set_group(0, "group_main")
    state.set_group(1, "group_main")
    state.set_position(0, {"x": 10.0, "y": 15.0})
    state.set_position(1, {"x": 20.0, "y": 25.0})
    state.set_position(2, {"x": 30.0, "y": 35.0})

    state.CZ(0, 1)
    state.CZ(1, 2)

    elements = state.to_cytoscape_elements()

    # Verify group element exists with value = None
    group_elems = [e for e in elements if e.get("data", {}).get("id") == "group_main"]
    assert len(group_elems) == 1
    assert isinstance(group_elems[0]["data"]["value"], str)

    # Reconstruct from elements
    reconstructed = FrontendClusterState.from_cytoscape_elements(elements)

    assert len(reconstructed) == 3
    assert reconstructed.get_position(0) == {"x": 10.0, "y": 15.0}
    assert reconstructed.get_group(0) == "group_main"
    assert reconstructed.get_group(1) == "group_main"
    assert reconstructed.get_group(2) is None
    assert reconstructed.cluster_state == state.cluster_state


def test_json_serialization():
    state = FrontendClusterState(3)
    state.add_group("g1", label="Group 1")
    state.set_group(0, "g1")
    state.set_position(0, {"x": 5.0, "y": 5.0})

    json_str = state.to_json()
    reconstructed = FrontendClusterState.from_json(json_str)

    assert reconstructed == state


def test_remove_node_by_position_id():
    state = FrontendClusterState(3)
    state.set_position(0, {"x": 0.0, "y": 0.0})
    state.set_position(1, {"x": 100.0, "y": 100.0})
    state.set_position(2, {"x": 200.0, "y": 200.0})

    # User selects node 1 on screen (position 100, 100) and deletes it
    removed = state.remove_node("1")
    assert removed == [1]
    assert len(state) == 2

    # Verify exported elements:
    # Cytoscape ID "0" has pos (0, 0)
    # Cytoscape ID "1" has pos (200, 200) and value=2
    elements = state.to_cytoscape_elements()
    n0 = next(e for e in elements if e.get("data", {}).get("id") in (0, "0"))
    n1 = next(e for e in elements if e.get("data", {}).get("id") in (1, "1"))

    assert n0["position"] == {"x": 0.0, "y": 0.0}
    assert n1["position"] == {"x": 200.0, "y": 200.0}
    assert isinstance(n1["data"]["value"], str)

    # User highlights node at position (200, 200) which has Cytoscape ID "1" and deletes it
    removed2 = state.remove_node("1")
    assert removed2 == [1]
    assert len(state) == 1

    # Remaining node is at (0, 0)
    elements2 = state.to_cytoscape_elements()
    assert len(elements2) == 1
    assert elements2[0]["position"] == {"x": 0.0, "y": 0.0}


def test_group_size_update_on_node_removal():
    state = FrontendClusterState(4)
    state.add_group("group_A", label="Group A")
    state.set_group(1, "group_A")
    state.set_group(2, "group_A")

    # Initial check: group_A has 2 children
    elems = state.to_cytoscape_elements()
    grp_children = [e for e in elems if e.get("data", {}).get("parent") == "group_A"]
    assert len(grp_children) == 2

    # 1. Remove Node 0 (outside group_A) -> group size remains 2
    state.remove_node("0")
    elems1 = state.to_cytoscape_elements()
    grp_children1 = [e for e in elems1 if e.get("data", {}).get("parent") == "group_A"]
    assert len(grp_children1) == 2

    # 2. Reconstruct from elems1 and remove Node 0 (inside group_A) -> group size decreases by 1
    state2 = FrontendClusterState.from_cytoscape_elements(elems1)
    state2.remove_node("0")
    elems2 = state2.to_cytoscape_elements()
    grp_children2 = [e for e in elems2 if e.get("data", {}).get("parent") == "group_A"]
    assert len(grp_children2) == 1


def test_quantum_operations_with_string_identifiers():
    state = FrontendClusterState(3)
    state.H("0")
    state.H("1")
    state.CZ("0", "1")
    state.toggle_edge("0", "1")
    state.add_edge("1", "2")
    state.remove_edge("1", "2")
    res = state.MZ("2", force=1)
    assert res == 0


def test_explicit_parent_none_in_cytoscape_elements():
    state = FrontendClusterState(2)
    state.add_group("g1")
    state.set_group(1, "g1")

    elems = state.to_cytoscape_elements()
    n0 = next(e for e in elems if e.get("data", {}).get("id") in (0, "0"))
    n1 = next(e for e in elems if e.get("data", {}).get("id") in (1, "1"))

    assert "parent" in n0["data"]
    assert n0["data"]["parent"] is None
    assert n1["data"]["parent"] == "g1"


def test_undo_group_creation():
    state = FrontendClusterState(2)
    state.set_position(0, {"x": 0.0, "y": 0.0})
    state.set_position(1, {"x": 100.0, "y": 100.0})

    # Save state before creating group
    state.push_state()

    # Action: Create group for node 1
    state.add_group("group_123", label="My Group")
    state.set_group(1, "group_123")

    # Verify group exists
    assert state.get_group(1) == "group_123"
    assert "group_123" in state.groups

    # Undo group creation
    assert state.undo() is True

    # Group should be undone!
    assert state.get_group(1) is None
    assert "group_123" not in state.groups
    assert len(state) == 2
