from clustersim.simulator import ClusterState


def test_apply_operation_wrapper_remove_last_node():
    from components.components_2d.action_panel import apply_operation_wrapper

    g = ClusterState(1)
    cyto_data = g.to_cytoscape(export_elements=True)
    for item in cyto_data:
        if item.get("data") and item["data"].get("vop"):
            item["position"] = {"x": 10, "y": 20}

    res = apply_operation_wrapper(
        method_name="remove_node",
        selected_nodes=[0],
        cyto_data=cyto_data,
        log="",
        method_args={},
        history_data={"undo_stack": [], "redo_stack": []},
    )

    assert len(res) == 5
    ui, cyto_data_new, repr_g, log, history_data = res
    assert ui == "Removed nodes [0]"
    assert cyto_data_new == []


def test_apply_operation_wrapper_errors():
    from components.components_2d.action_panel import apply_operation_wrapper
    from dash import no_update

    g = ClusterState(2)
    cyto_data = g.to_cytoscape(export_elements=True)
    for item in cyto_data:
        if item.get("data") and item["data"].get("vop"):
            item["position"] = {"x": 10, "y": 20}

    res_cz = apply_operation_wrapper(
        method_name="CZ",
        selected_nodes=[0],
        cyto_data=cyto_data,
        log="",
        method_args={},
        history_data={"undo_stack": [], "redo_stack": []},
    )
    assert len(res_cz) == 5
    assert res_cz[0] == "Error: Odd number of gates!"
    assert res_cz[1:] == (no_update, no_update, no_update, no_update)

    res_fusion = apply_operation_wrapper(
        method_name="fusion_gate",
        selected_nodes=[0],
        cyto_data=cyto_data,
        log="",
        method_args={"fusion_type": "success", "force": 0, "mode": "success"},
        history_data={"undo_stack": [], "redo_stack": []},
    )
    assert len(res_fusion) == 5
    assert res_fusion[0] == "Error: Odd number of qubits!"
    assert res_fusion[1:] == (no_update, no_update, no_update, no_update)


def test_apply_operation_wrapper_add_node_offset():
    from components.components_2d.action_panel import apply_operation_wrapper

    # 1. No selected nodes: fallback to random position
    g = ClusterState(2)
    cyto_data = g.to_cytoscape(export_elements=True)
    for item in cyto_data:
        if item.get("data") and item["data"].get("vop"):
            item["position"] = {"x": 10, "y": 20}

    res = apply_operation_wrapper(
        method_name="add_node",
        selected_nodes=[],
        cyto_data=cyto_data,
        log="",
        method_args={},
        history_data={"undo_stack": [], "redo_stack": []},
    )
    assert len(res) == 5
    ui, cyto_data_new, repr_g, log, history_data = res
    assert ui == "Added node 2!"
    new_node_pos = next(
        item["position"]
        for item in cyto_data_new
        if item.get("data") and str(item["data"].get("id")) == "2"
    )
    assert 0 <= new_node_pos["x"] <= 100
    assert 0 <= new_node_pos["y"] <= 100

    # 2. Selected nodes: offset by +50 from centroid of selected nodes
    g2 = ClusterState(2)
    cyto_data2 = g2.to_cytoscape(export_elements=True)
    cyto_data2[0]["position"] = {"x": 10, "y": 20}
    cyto_data2[1]["position"] = {"x": 30, "y": 40}

    res2 = apply_operation_wrapper(
        method_name="add_node",
        selected_nodes=[0, 1],
        cyto_data=cyto_data2,
        log="",
        method_args={},
        history_data={"undo_stack": [], "redo_stack": []},
    )
    assert len(res2) == 5
    ui2, cyto_data_new2, repr_g2, log2, history_data2 = res2
    assert ui2 == "Added node 2 with edges to [0, 1]!"
    new_node_pos2 = next(
        item["position"]
        for item in cyto_data_new2
        if item.get("data") and str(item["data"].get("id")) == "2"
    )
    assert new_node_pos2 == {"x": 70, "y": 80}


def test_apply_operation_wrapper_duplicate():
    from components.components_2d.action_panel import apply_operation_wrapper
    from clustersim.simulator import FrontendClusterState

    # 1. Setup a FrontendClusterState with 3 nodes, two of which are in a group
    state = FrontendClusterState(3)
    state.add_group("group_alpha", label="Alpha Group")
    state.set_group(0, "group_alpha")
    state.set_group(1, "group_alpha")
    
    state.set_position(0, {"x": 10.0, "y": 20.0})
    state.set_position(1, {"x": 30.0, "y": 40.0})
    state.set_position(2, {"x": 50.0, "y": 60.0})
    
    state.CZ(0, 1)

    cyto_data = state.to_cytoscape_elements()

    # 2. Duplicate node 0 (which is in group_alpha)
    res = apply_operation_wrapper(
        method_name="duplicate",
        selected_nodes=["0"],
        cyto_data=cyto_data,
        log="",
        method_args={},
        history_data={"undo_stack": [], "redo_stack": []},
    )
    
    ui, cyto_data_new, repr_g, log_out, history_data_out = res
    assert ui == "Duplicated nodes [0]"
    
    # 3. Reconstruct to check metadata and counts
    reconstructed = FrontendClusterState.from_cytoscape_elements(cyto_data_new)
    assert len(reconstructed) == 4 # Original 3 + 1 duplicated
    
    assert reconstructed.get_position(3) == {"x": 60.0, "y": 70.0}
    new_group_id = reconstructed.get_group(3)
    assert new_group_id is not None
    assert new_group_id != "group_alpha"
    assert new_group_id in reconstructed.groups
    assert reconstructed.groups[new_group_id]["label"] == "Alpha Group"
    
    # 4. Duplicate the whole group "group_alpha"
    res_group = apply_operation_wrapper(
        method_name="duplicate",
        selected_nodes=["group_alpha"],
        cyto_data=cyto_data_new,
        log="",
        method_args={},
        history_data={"undo_stack": [], "redo_stack": []},
    )
    ui_g, cyto_data_group, _, _, _ = res_group
    
    reconstructed_g = FrontendClusterState.from_cytoscape_elements(cyto_data_group)
    # The group contains 0 and 1. So duplication adds 2 nodes.
    # Total nodes should be 4 (from cyto_data_new) + 2 = 6.
    assert len(reconstructed_g) == 6
    
    parent_4 = reconstructed_g.get_group(4)
    parent_5 = reconstructed_g.get_group(5)
    
    assert parent_4 is not None
    assert parent_4 == parent_5
    assert parent_4 != "group_alpha"
    assert parent_4 in reconstructed_g.groups
    assert reconstructed_g.groups[parent_4]["label"] == "Alpha Group"


def test_copy_nodes_with_group():
    from components.components_2d.action_panel import copy_nodes
    from clustersim.simulator import FrontendClusterState

    # 1. Setup a FrontendClusterState with 2 nodes inside a group
    state = FrontendClusterState(2)
    state.add_group("group_alpha", label="Alpha Group")
    state.set_group(0, "group_alpha")
    state.set_group(1, "group_alpha")
    
    cyto_data = state.to_cytoscape_elements()
    
    # 2. Case A: Only the group is selected
    selected_node_data_group = [
        {"id": "group_alpha", "value": state.groups["group_alpha"]["value"]}
    ]
    
    clipboard = copy_nodes(1, selected_node_data_group, cyto_data)
    assert len(clipboard) == 2
    assert state.get_value(0) in clipboard
    assert state.get_value(1) in clipboard

    # 3. Case B: The group AND its children are selected (should only copy/duplicate once)
    selected_node_data_both = [
        {"id": "group_alpha", "value": state.groups["group_alpha"]["value"]},
        {"id": "0", "value": state.get_value(0)},
        {"id": "1", "value": state.get_value(1)},
    ]
    
    clipboard_both = copy_nodes(1, selected_node_data_both, cyto_data)
    assert len(clipboard_both) == 2
    assert state.get_value(0) in clipboard_both
    assert state.get_value(1) in clipboard_both


def test_action_panel_tooltips():
    from components.components_2d.action_panel import qubit_panel
    import dash_bootstrap_components as dbc

    tooltips = [
        child for child in qubit_panel.children if isinstance(child, dbc.Tooltip)
    ]

    expected_targets = {
        "duplicate",
        "remove-node",
        "add-node",
        "toggle-edge",
        "remove-edge",
        "H",
        "LC",
        "fusion-gate-xxzz",
        "fusion-gate-xzzx",
        "fusion-gate-xyyz",
        "X",
        "Y",
        "Z",
        "S",
        "S_DAG",
        "CX",
        "CZ",
        "LCR",
        "MX",
        "MY",
        "MZ",
    }

    actual_targets = {tooltip.target for tooltip in tooltips}
    assert expected_targets == actual_targets


def test_fusion_buttons_mapping():
    from components.components_2d.action_panel import button_operations

    assert "fusion-gate" not in button_operations
    assert "fusion-gate-xxzz" in button_operations
    assert "fusion-gate-xzzx" in button_operations
    assert "fusion-gate-xyyz" in button_operations

    for key, expected_type in [
        ("fusion-gate-xxzz", "XXZZ"),
        ("fusion-gate-xzzx", "XZZX"),
        ("fusion-gate-xyyz", "XYYZ"),
    ]:
        op_name, op_args = button_operations[key]
        assert op_name == "fusion_gate"
        assert op_args["fusion_type"] == expected_type


def test_load_graph_callback_json():
    from components.components_2d.action_panel import load_graph
    from clustersim.simulator import FrontendClusterState
    from unittest.mock import patch
    import json
    import base64

    # Create dummy initial state data
    initial_state = FrontendClusterState(2)
    initial_elements = initial_state.to_cytoscape_elements()
    initial_log = "original log content"
    initial_history = {"undo_stack": [], "redo_stack": []}

    # Target state we want to load
    target_state = FrontendClusterState(3)
    target_state.node_metadata[0]["shape"] = "star"
    target_state.set_position(1, {"x": 100.0, "y": 200.0})
    target_json_str = target_state.to_json()

    # 1. Test loading via JSON text input box
    with patch("components.components_2d.action_panel.callback_context") as mock_ctx:
        mock_ctx.triggered_id = "load-button"
        res = load_graph(
            _undo=None,
            _ctrl_z=None,
            _redo=None,
            _ctrl_y=None,
            _load=1,
            _reset=None,
            upload_contents=None,
            upload_filename=None,
            load_graph_input=target_json_str,
            move_log=initial_log,
            cyto_data=initial_elements,
            history_data=initial_history.copy(),
        )
        assert len(res) == 9
        cyto_data_new, repr_g, new_move_log, ui_msg, color, history_data_new, valid, invalid, feedback = res
        assert valid is True
        assert invalid is False
        assert ui_msg == "Action completed!"
        # Check that history_data has pushed the prior state onto the undo stack!
        assert len(history_data_new["undo_stack"]) == 1
        assert history_data_new["undo_stack"][0]["move_log"] == initial_log
        assert history_data_new["undo_stack"][0]["elements"] == initial_elements

    # 2. Test loading via file upload (dcc.Upload)
    upload_contents_b64 = "data:application/json;base64," + base64.b64encode(target_json_str.encode("utf-8")).decode("utf-8")
    with patch("components.components_2d.action_panel.callback_context") as mock_ctx:
        mock_ctx.triggered_id = "upload-graph-json"
        res = load_graph(
            _undo=None,
            _ctrl_z=None,
            _redo=None,
            _ctrl_y=None,
            _load=None,
            _reset=None,
            upload_contents=upload_contents_b64,
            upload_filename="test_graph.json",
            load_graph_input="",
            move_log=initial_log,
            cyto_data=initial_elements,
            history_data=initial_history.copy(),
        )
        assert len(res) == 9
        cyto_data_new, repr_g, new_move_log, ui_msg, color, history_data_new, valid, invalid, feedback = res
        assert valid is True
        assert invalid is False
        assert "Successfully loaded test_graph.json" in ui_msg
        assert len(history_data_new["undo_stack"]) == 1
        assert history_data_new["undo_stack"][0]["move_log"] == initial_log
        assert history_data_new["undo_stack"][0]["elements"] == initial_elements


def test_load_graph_exception_handling():
    from components.components_2d.action_panel import load_graph
    from clustersim.simulator import FrontendClusterState
    from unittest.mock import patch
    import base64

    initial_state = FrontendClusterState(2)
    initial_elements = initial_state.to_cytoscape_elements()
    initial_log = "original log content"
    initial_history = {"undo_stack": [], "redo_stack": []}

    # 1. Test empty load input (should succeed and result in a valid empty graph)
    with patch("components.components_2d.action_panel.callback_context") as mock_ctx:
        mock_ctx.triggered_id = "load-button"
        res = load_graph(
            _undo=None,
            _ctrl_z=None,
            _redo=None,
            _ctrl_y=None,
            _load=1,
            _reset=None,
            upload_contents=None,
            upload_filename=None,
            load_graph_input="",
            move_log=initial_log,
            cyto_data=initial_elements,
            history_data=initial_history.copy(),
        )
        cyto_data_new, repr_g, new_move_log, ui_msg, color, history_data_new, valid, invalid, feedback = res
        assert invalid is False
        assert valid is True
        assert cyto_data_new == []

    # 2. Test invalid/malformed JSON upload (triggers IndexError or parse error)
    # The JSON state has non-contiguous node IDs: 0, 2, 3, which induces dynamic expansion and IndexError in node_metadata.
    non_contiguous_json = (
        '{"cluster_state_json": "{\\"directed\\":false,\\"multigraph\\":false,\\"attrs\\":null,'
        '\\"nodes\\":[{\\"id\\":0,\\"data\\":{\\"id\\":\\"0\\",\\"vop\\":\\"IA\\"}},'
        '{\\"id\\":2,\\"data\\":{\\"id\\":\\"1\\",\\"vop\\":\\"IA\\"}},'
        '{\\"id\\":3,\\"data\\":{\\"id\\":\\"2\\",\\"vop\\":\\"IA\\"}}],\\"links\\":[]}",'
        '"node_metadata": [{}, {}, {}], "groups": {}}'
    )
    upload_contents_b64 = "data:application/json;base64," + base64.b64encode(non_contiguous_json.encode("utf-8")).decode("utf-8")
    with patch("components.components_2d.action_panel.callback_context") as mock_ctx:
        mock_ctx.triggered_id = "upload-graph-json"
        res = load_graph(
            _undo=None,
            _ctrl_z=None,
            _redo=None,
            _ctrl_y=None,
            _load=None,
            _reset=None,
            upload_contents=upload_contents_b64,
            upload_filename="test_graph.json",
            load_graph_input="",
            move_log=initial_log,
            cyto_data=initial_elements,
            history_data=initial_history.copy(),
        )
        cyto_data_new, repr_g, new_move_log, ui_msg, color, history_data_new, valid, invalid, feedback = res
        assert invalid is True
        assert valid is False
        assert "Failed to load file" in ui_msg
        assert "list index out of range" in feedback


def test_handle_buttons_edge_grouped_nodes():
    from components.components_2d.action_panel import handle_buttons, button_operations, KEY_MAPPINGS
    from clustersim.simulator import FrontendClusterState
    from unittest.mock import patch

    state = FrontendClusterState(3)
    state.add_group("group_alpha", label="Alpha Group")
    state.set_group(0, "group_alpha")
    state.set_group(1, "group_alpha")

    cyto_data = state.to_cytoscape_elements()

    # Edge between grouped nodes: '0_group_alpha' and '1_group_alpha'
    selected_edge_data = [
        {"source": "0_group_alpha", "target": "1_group_alpha"}
    ]

    with patch("components.components_2d.action_panel.callback_context") as mock_ctx:
        mock_ctx.triggered_id = "toggle-edge"
        num_inputs = len(button_operations) + len(KEY_MAPPINGS) + 1
        args = [0] * num_inputs + [
            [],  # clipboard_data
            {"undo_stack": [], "redo_stack": []},  # history_data
            "success",  # fusion_mode_val
            0,  # fusion_force_val
            0,  # force_meas_val
            "",  # log
            [],  # selected_node_data (empty, so it falls back to edge)
            selected_edge_data,
            cyto_data,
        ]

        res = handle_buttons(*args)
        ui, color, cyto_data_new, repr_g, log, history_data = res
        assert "Applied toggle_edge" in ui


def test_load_graph_reset_button():
    from components.components_2d.action_panel import load_graph
    from clustersim.simulator import FrontendClusterState
    from unittest.mock import patch

    initial_state = FrontendClusterState(2)
    initial_elements = initial_state.to_cytoscape_elements()
    initial_log = "original log content"
    initial_history = {"undo_stack": [], "redo_stack": []}

    # Test loading via reset-button triggers load of empty graph
    with patch("components.components_2d.action_panel.callback_context") as mock_ctx:
        mock_ctx.triggered_id = "reset-button"
        res = load_graph(
            _undo=None,
            _ctrl_z=None,
            _redo=None,
            _ctrl_y=None,
            _load=None,
            _reset=1,
            upload_contents=None,
            upload_filename=None,
            load_graph_input="some previous text input",  # Should be ignored and treated as ""
            move_log=initial_log,
            cyto_data=initial_elements,
            history_data=initial_history.copy(),
        )
        cyto_data_new, repr_g, new_move_log, ui_msg, color, history_data_new, valid, invalid, feedback = res
        assert invalid is False
        assert valid is True
        assert cyto_data_new == []


def test_move_log_uses_ids_not_values():
    from components.components_2d.action_panel import handle_buttons, button_operations, KEY_MAPPINGS
    from clustersim.simulator import FrontendClusterState
    from unittest.mock import patch

    state = FrontendClusterState(3)
    state.set_position(0, {"x": 10, "y": 20})
    state.set_position(1, {"x": 30, "y": 40})
    state.set_position(2, {"x": 50, "y": 60})

    v0 = state.node_metadata[0]["value"]
    v1 = state.node_metadata[1]["value"]

    cyto_data = state.to_cytoscape_elements()

    selected_node_data = [
        {"id": "0", "value": v0},
        {"id": "1", "value": v1},
    ]

    with patch("components.components_2d.action_panel.callback_context") as mock_ctx:
        mock_ctx.triggered_id = "CZ"
        num_inputs = len(button_operations) + len(KEY_MAPPINGS) + 1
        args = [0] * num_inputs + [
            [],  # clipboard_data
            {"undo_stack": [], "redo_stack": []},  # history_data
            "success",  # fusion_mode_val
            0,  # fusion_force_val
            0,  # force_meas_val
            "",  # log
            selected_node_data,
            [],  # selected_edge_data
            cyto_data,
        ]

        res = handle_buttons(*args)
        ui, color, cyto_data_new, repr_g, log, history_data = res
        assert "CZ 0 1\n" in log
        assert v0 not in log
        assert v1 not in log





