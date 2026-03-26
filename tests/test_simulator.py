from cluster_sim.simulator import ClusterState
import rustworkx as rx
import networkx as nx


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

    g, log = ClusterState.load_text(text, return_log=True)

    assert log == text
    assert str(g) == "['+ZZZXY', '-XXZXY', '-XZXXI', '-XZZZI', '+XZIIZ']"


def test_load_text_edge():
    text = "H 0 1 2 3 4\nCZ 0 1 1 2 2 3 3 4 4 0\nADD_EDGE 3 4 0 2 1\n"

    g, log = ClusterState.load_text(text, return_log=True)

    assert log == text
    assert str(g) == "['+XZZZZ', '+ZXZZZ', '+ZZXZZ', '+ZZZXZ', '+ZZZZX']"


def test_reduced_form():
    c = ClusterState(24)
    for i in range(24):
        c.H(i)

    local_clifford = {
        'IA': 'I',
        'XA': 'X', # HSSH
        'YA': 'Y', # SSHSSH
        'ZA': 'Z', # SS
        'IB': 'HSSHS', 
        'XB': 'SSS',
        'YB': 'S',
        'ZB': 'SSHSSHS',
        'IC': 'HSSHSSH',
        'XC': 'HSS',
        'YC': 'H',
        'ZC': 'SSH',
        'ID': 'SSSHS',
        'XD': 'SSHSH',
        'YD': 'SHS',
        'ZD': 'HSH',
        'IE': 'SHSH',
        'XE': 'SSHS',
        'YE': 'SSSHSH',
        'ZE': 'HS',
        'IF': 'SH',
        'YF': 'SHSSHSSH',
        'XF': 'SSSH',
        'ZF': 'SHSS',
    }

    for i, vop in enumerate(local_clifford):
        c.apply_VOP(i, vop=vop)

    for i, j in zip(c.stabilizers, c.reduced_form().stabilizers):
        assert i == j


def test_local_clifford_table():
    c = ClusterState(24)
    c2 = ClusterState(24)

    for i in range(24):
        c.H(i)
        c2.H(i)

    local_clifford = {
        'IA': 'I',
        'XA': 'X', # HSSH
        'YA': 'Y', # SSHSSH
        'ZA': 'Z', # SS
        'IB': 'HSSHS', 
        'XB': 'SSS',
        'YB': 'S',
        'ZB': 'SSHSSHS',
        'IC': 'HSSHSSH',
        'XC': 'HSS',
        'YC': 'H',
        'ZC': 'SSH',
        'ID': 'SSSHS',
        'XD': 'SSHSH',
        'YD': 'SHS',
        'ZD': 'HSH',
        'IE': 'SHSH',
        'XE': 'SSHS',
        'YE': 'SSSHSH',
        'ZE': 'HS',
        'IF': 'SH',
        'YF': 'SHSSHSSH',
        'XF': 'SSSH',
        'ZF': 'SHSS',
    }

    for i, (vop, decomposition) in enumerate(local_clifford.items()):
        c.apply_VOP(i, vop=vop)
        for gate in decomposition[::-1]:
            getattr(c2, gate)(i)
        

    for i, j in zip(c.stabilizers, c2.stabilizers):
        assert i == j