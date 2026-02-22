from rustworkx.visualization import mpl_draw
from graphsim import GraphRegister
import rustworkx as rx


class ClusterState:
    """
    Adapter class containing a GraphRegister object.
    """

    def __init__(self, num_nodes: int):
        """
        Nodes are indexed from 0 to n-1, where n is the number of nodes in the graph.

        Args:
            graph (rx.PyGraph | rx.PyDiGraph): A rustworkx graph object.
            init_graph_state (bool, optional): Whether to auto apply H and CZ. Defaults to False.
        """

        self.num_nodes = num_nodes
        self.graph_state = GraphRegister(self.num_nodes)

        # if any(not isinstance(n, int) for n in graph.nodes):
        #     self.graph = rx.convert_node_labels_to_integers(graph)
        # else:
        #     self.graph = graph

    def __repr__(self):
        rep = "Adjacency List:\n"
        rep += self.graph_state.print_adjacency_list()
        rep += "Stabilizers:\n"
        rep += self.graph_state.print_stabilizers()
        return rep

    def __str__(self):
        return self.graph_state.print_stabilizers()

    def measure(self, qubit: int, basis: str):
        """
        Measure a node in the graph state.

        Args:
            qubit: The qubit to measure, which is an integer from 0 to n-1.
            basis: The basis to measure in, which is either 'X', 'Y' or 'Z'.
        """

        if basis == "Z":
            pass
        elif basis == "X":
            self.H(qubit)
        elif basis == "Y":
            self.S_DAG(qubit)
            self.H(qubit)

        return self.graph_state.measure(qubit)

    def X(self, qubit: int):
        self.graph_state.X(qubit)

    def Y(self, qubit: int):
        self.graph_state.Y(qubit)

    def Z(self, qubit: int):
        self.graph_state.Z(qubit)

    def H(self, qubit: int):
        self.graph_state.H(qubit)

    def S(self, qubit: int):
        self.graph_state.S(qubit)

    def S_DAG(self, qubit: int):
        self.graph_state.S(qubit)
        self.graph_state.S(qubit)
        self.graph_state.S(qubit)

    def CX(self, control: int, qubit: int):
        self.graph_state.CX(control, qubit)

    def CZ(self, control: int, qubit: int):
        self.graph_state.CZ(control, qubit)

    def LC(self, qubit):
        raise NotImplementedError
        self.graph_state.local_complementation(qubit)
        self.graph = self.graph_state.to_networkx()

    @classmethod
    def from_json(cls, json_data):
        """
        Convert the graph state to a JSON-serializable format.

        Returns:
            A JSON-serializable representation of the graph state.
        """

        cls.from_rustworkx(rx.parse_node_link_json(json_data))
        cls._sync_graph()

        return cls

    def to_json(self):
        """
        Convert the graph state to a JSON-serializable format.

        Returns:
            A JSON-serializable representation of the graph state.
        """
        return rx.node_link_json(self.to_rustworkx(), node_attrs= lambda node: node)

    @property
    def stabilizers(self):
        return self.graph_state.stabilizer_list()

    @property
    def adjacency_matrix(self):
        return self.graph_state.adjacency_matrix()

    @property
    def adjacency_list(self):
        return self.graph_state.adjacency_list()

    @property
    def vertex_operators(self):
        return self.graph_state.vop_list()

    def _sync_graph(self):
        """This function should apply the VOPs to a graph state.
        """

        for vop in self.vertex_operators:
            pass
            self.graph_state

        raise NotImplementedError

    def to_rustworkx(self):
        """Export data from the underlying state.

        Returns:
            rx.PyGraph: rustworkx representation
        """

        g = rx.PyGraph(multigraph=False)

        g.add_nodes_from(range(self.num_nodes))

        g.add_edges_from([edge for edge in self._edge_list_from_adjacency_list()])

        for node_index in g.node_indices():
            g[node_index] = {
                "stabilizer": self.stabilizers[node_index],
                "vop": self.vertex_operators[node_index],
            }

        return g

    def _edge_list_from_adjacency_list(self):
        for u, neighbors in enumerate(self.adjacency_list):
            for v in neighbors:
                if u < v:
                    yield (u, v, None)

    @classmethod
    def from_rustworkx(cls, graph: rx.PyGraph | rx.PyDiGraph):
        """Create a graph state from rustworkx.

        Args:
            graph (rx.PyGraph | rx.PyDiGraph): rustworkx graph.

        Returns:
            ClusterState : class object
        """
        c = cls(graph.num_nodes())
        for i in graph.node_indices():
            c.H(i)

        for e in graph.edge_list():
            c.CZ(e[0], e[1])

        return c

    def draw(self, label_func = lambda node: str(node), **kwargs):
        g = self.to_rustworkx()

        mpl_draw(g, with_labels=True, labels=label_func, **kwargs)
