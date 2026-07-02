from cluster_sim.simulator import ClusterState
import rustworkx as rx
import networkx as nx
import pytest


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


def test_import_export_json():

    g = rx.generators.grid_graph(3, 3)
    c = ClusterState.from_rustworkx(g)
    c.measure(4, basis="Y")
    d = c.to_json()

    c2 = ClusterState.from_json(d)

    assert c == c2


def test_import_export_networkx():
    g = nx.grid_2d_graph(3, 2)
    c = ClusterState.from_networkx(g)
    c.measure(4, basis="Y")
    d = c.to_networkx()

    c2 = ClusterState.from_networkx(d)
    assert c == c2


def test_import_export_cytoscape():
    g = nx.grid_2d_graph(3, 2)

    c = ClusterState.from_networkx(g)
    c.H(0)
    c.measure(4, basis="Y")
    d = c.to_cytoscape()
    c2 = ClusterState.from_cytoscape(d)
    d2 = c2.to_cytoscape()
    c3 = ClusterState.from_cytoscape(d2)
    assert c3 == c2


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


def test_from_json_formats(tmp_path):
    import io
    import json

    g = rx.generators.grid_graph(2, 2)
    c = ClusterState.from_rustworkx(g)
    json_str = c.to_json()
    json_dict = json.loads(json_str)

    # 1. Test dictionary
    c_dict = ClusterState.from_json(json_dict)
    assert c == c_dict

    # 2. Test string
    c_str = ClusterState.from_json(json_str)
    assert c == c_str

    # 3. Test file path (str)
    file_path_str = str(tmp_path / "graph.json")
    with open(file_path_str, "w") as f:
        f.write(json_str)
    c_path_str = ClusterState.from_json(file_path_str)
    assert c == c_path_str

    # 4. Test path-like object (pathlib.Path)
    file_path = tmp_path / "graph_path.json"
    file_path.write_text(json_str)
    c_path = ClusterState.from_json(file_path)
    assert c == c_path

    # 5. Test file-like object
    string_io = io.StringIO(json_str)
    c_file = ClusterState.from_json(string_io)
    assert c == c_file

    # 6. Test FileNotFoundError
    with pytest.raises(FileNotFoundError):
        ClusterState.from_json("non_existent_file.json")

    # 7. Test TypeError
    with pytest.raises(TypeError):
        ClusterState.from_json(12345)
