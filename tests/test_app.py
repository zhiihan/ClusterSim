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
    assert res_cz[0] == "Odd number of gates!"
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
    assert res_fusion[0] == "Odd number of qubits!"
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
            upload_contents=None,
            upload_filename=None,
            load_graph_input=target_json_str,
            move_log=initial_log,
            cyto_data=initial_elements,
            history_data=initial_history.copy(),
        )
        assert len(res) == 8
        cyto_data_new, repr_g, new_move_log, ui_msg, history_data_new, valid, invalid, feedback = res
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
            upload_contents=upload_contents_b64,
            upload_filename="test_graph.json",
            load_graph_input="",
            move_log=initial_log,
            cyto_data=initial_elements,
            history_data=initial_history.copy(),
        )
        assert len(res) == 8
        cyto_data_new, repr_g, new_move_log, ui_msg, history_data_new, valid, invalid, feedback = res
        assert valid is True
        assert invalid is False
        assert "Successfully loaded test_graph.json" in ui_msg
        assert len(history_data_new["undo_stack"]) == 1
        assert history_data_new["undo_stack"][0]["move_log"] == initial_log
        assert history_data_new["undo_stack"][0]["elements"] == initial_elements

