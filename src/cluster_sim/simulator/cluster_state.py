from cluster_sim.graph_state import GraphState
import rustworkx as rx
import networkx as nx


class ClusterState:
    """
    Adapter class containing a GraphState object.
    """

    def __init__(self, graph: nx.Graph | rx.PyGraph = rx.PyGraph()):
        """
        Create a ClusterState object from a NetworkX graph.

        Attributes:
            graph: A networkx graph object.
            graph_state: A GraphState object.

        Notes:
            Nodes are indexed from 0 to n-1, where n is the number of nodes in the graph.
        """

        if isinstance(graph, nx.Graph):
            self.graph = rx.networkx_converter(graph)
            for node in self.graph.node_indices():
                self.graph[node] = {"index": node}
            self.graph_state = GraphState(len(graph.nodes))
        elif isinstance(graph, rx.PyGraph):
            self.graph = graph
            for node in self.graph.node_indices():
                if self.graph[node] is None:
                    self.graph[node] = {"index": node}
            self.graph_state = GraphState(graph.num_nodes())
        else:
            raise TypeError("Graph must be a NetworkX graph or a RustworkX PyGraph.")

        for i in self.graph.node_indices():
            self.graph_state.h(i)

        for e in self.graph.edge_indices():
            edge_data = self.graph.get_edge_endpoints_by_index(e)
            self.graph_state.add_edge(edge_data[0], edge_data[1])

    def measure(self, qubit: int, basis: str):
        """
        Measure a node in the graph state.

        Args:
            qubit: The qubit to measure, which is an integer from 0 to n-1.
            basis: The basis to measure in, which is either 'X', 'Y' or 'Z'.
        """

        return self.graph_state.measure(qubit, basis=basis)

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

    @classmethod
    def from_json(cls, json_data):
        """
        Convert the graph state to a JSON-serializable format.

        Returns:
            A JSON-serializable representation of the graph state.
        """
        return cls(
            rx.parse_node_link_json(
                json_data, node_attrs=lambda node: {"index": int(node["index"])}
            )
        )

    def to_json(self):
        """
        Convert the graph state to a JSON-serializable format.

        Returns:
            A JSON-serializable representation of the graph state.
        """
        return rx.node_link_json(
            self.graph, node_attrs=lambda node: {"index": str(node["index"])}
        )


class RustworkXState:
    """
    Adapter class containing a RustworkX graph object.
    """

    def __init__(self, graph: nx.Graph | rx.PyGraph = rx.PyGraph()):
        """
        Nodes are indexed from 0 to n-1, where n is the number of nodes in the graph.

        Attributes:
            graph: A rustworkx graph object.
        """
        self.graph = graph
        for n in self.graph.node_indices():
            self.graph[n] = n

    def sync_graph(self):
        pass

    @classmethod
    def from_json(cls, json_data):
        """
        Convert the graph state to a JSON-serializable format.

        Returns:
            A JSON-serializable representation of the graph state.
        """
        return cls(rx.parse_node_link_json(json_data))

    def to_json(self):
        """
        Convert the graph state to a JSON-serializable format.

        Returns:
            A JSON-serializable representation of the graph state.
        """
        return rx.node_link_json(self.graph)

    def add_node(self, *args, **kwargs):
        """
        Add a node to the graph.

        Args:
            args: Positional arguments for the add_node method.
            kwargs: Keyword arguments for the add_node method.
        """
        self.graph.add_node(*args, **kwargs)
