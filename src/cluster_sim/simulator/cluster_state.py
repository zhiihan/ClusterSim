from cluster_sim.graph_state import GraphState
import networkx as nx


class ClusterState:
    """
    Adapter class containing a GraphState object.
    """

    def __init__(self, graph: nx.Graph):
        """
        Nodes are indexed from 0 to n-1, where n is the number of nodes in the graph.

        Attributes:
            graph: A networkx graph object.
            graph_state: A GraphState object.
        """

        self.graph_state = GraphState(len(graph.nodes))

        if any(not isinstance(n, int) for n in graph.nodes):
            self.graph = nx.convert_node_labels_to_integers(graph)
        else:
            self.graph = graph

        for i in self.graph.nodes:
            self.graph_state.h(i)

        for e in self.graph.edges:
            self.graph_state.add_edge(*e)

    def measure(self, qubit: int, basis: str):
        """
        Measure a node in the graph state.

        Args:
            qubit: The qubit to measure, which is an integer from 0 to n-1.
            basis: The basis to measure in, which is either 'X', 'Y' or 'Z'.
        """
        self.graph_state.measure(qubit, basis=basis)
        self.graph = self.graph_state.to_networkx()

    def x(self, qubit: int):
        self.graph_state.x(qubit)

    def y(self, qubit: int):
        self.graph_state.y(qubit)

    def z(self, qubit: int):
        self.graph_state.z(qubit)

    def h(self, qubit: int):
        self.graph_state.h(qubit)

    def s(self, qubit: int):
        self.graph_state.s(qubit)

    def s_dagger(self, qubit: int):
        self.graph_state.s_dagger(qubit)

    def cx(self, control: int, qubit: int):
        self.graph_state.cx(control, qubit)

    def cz(self, control: int, qubit: int):
        self.graph_state.cz(control, qubit)

    def sync_graph(self):
        self.graph = self.graph_state.to_networkx()
        self.graph.remove_nodes_from(list(nx.isolates(self.graph)))

    @classmethod
    def from_json(cls, json_data):
        """
        Convert the graph state to a JSON-serializable format.

        Returns:
            A JSON-serializable representation of the graph state.
        """
        return cls(nx.node_link_graph(json_data, edges="edges"))

    def to_json(self):
        """
        Convert the graph state to a JSON-serializable format.

        Returns:
            A JSON-serializable representation of the graph state.
        """
        return nx.node_link_data(self.graph, edges="edges")


class NetworkXState:
    """
    Adapter class containing a NetworkX graph object.
    """

    def __init__(self, graph: nx.Graph):
        """
        Nodes are indexed from 0 to n-1, where n is the number of nodes in the graph.

        Attributes:
            graph: A networkx graph object.
        """
        self.graph = graph

    def sync_graph(self):
        pass

    def add_node(self, i):
        self.graph.add_node(i)

    @classmethod
    def from_json(cls, json_data):
        """
        Create a NetworkXState object from JSON data.

        Returns:
            A NetworkXState object.
        """
        return cls(nx.node_link_graph(json_data, edges="edges"))

    def to_json(self):
        """
        Convert the graph state to a JSON-serializable format.

        Returns:
            A JSON-serializable representation of the graph state.
        """
        return nx.node_link_data(self.graph, edges="edges")
