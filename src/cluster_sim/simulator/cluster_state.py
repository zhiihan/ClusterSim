from rustworkx.visualization import mpl_draw
import graphsim
import rustworkx as rx
import networkx as nx


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
        self.simulator = graphsim.GraphRegister(self.num_nodes)

    def __repr__(self):
        rep = "Adjacency List:\n"
        rep += self.simulator.print_adjacency_list()
        rep += "Stabilizers:\n"
        rep += str(self.stabilizers)
        return rep

    def __eq__(self, other):
        if not isinstance(other, ClusterState):
            raise NotImplementedError
        else:
            return (self.vertex_operators == other.vertex_operators) and (
                self.adjacency_list == other.adjacency_list
            )

    def __str__(self):
        return str(self.stabilizers)

    def __len__(self):
        return self.num_nodes

    def measure(self, qubit: int, force: int = -1, basis: str = "Z"):
        """
        Measure a node in the graph state.

        Args:
            qubit: The qubit to measure, which is an integer from 0 to n-1.
            basis: The basis to measure in, which is either 'X', 'Y' or 'Z'.
        """

        return self.simulator.measure(qubit, force, basis)

    def X(self, qubit: int):
        self.simulator.X(qubit)

    def Y(self, qubit: int):
        self.simulator.S(qubit)
        self.simulator.S(qubit)
        self.simulator.H(qubit)
        self.simulator.S(qubit)
        self.simulator.S(qubit)
        self.simulator.H(qubit)

    def Z(self, qubit: int):
        self.simulator.Z(qubit)

    def H(self, qubit: int):
        self.simulator.H(qubit)

    def S(self, qubit: int):
        self.simulator.S(qubit)

    def S_DAG(self, qubit: int):
        self.simulator.S(qubit)
        self.simulator.S(qubit)
        self.simulator.S(qubit)

    def CX(self, control: int, qubit: int):
        self.simulator.CX(control, qubit)

    def CZ(self, control: int, qubit: int):
        self.simulator.CZ(control, qubit)

    def local_complementation(self, qubit):
        """Apply a local complementation.

        Args:
            qubit (int): which qubit to apply
        """
        self.simulator.invert_neighborhood(qubit)

    def apply_VOP(self, qubit, vop: tuple[int, int] | int | str):
        """Apply vertex operators.

        Args:
            qubit (int): which vertex operator to apply
        """
        if isinstance(vop, tuple):
            self.simulator.VOP(qubit, graphsim.LocCliffOp(*vop))
        else:
            self.simulator.VOP(qubit, graphsim.LocCliffOp(vop))

    def add_edge(self, qubit1, qubit2):
        self.simulator.add_edge(qubit1, qubit2)

    def remove_edge(self, qubit1, qubit2):
        self.simulator.del_edge(qubit1, qubit2)

    def toggle_edge(self, qubit1, qubit2):
        self.simulator.toggle_edge(qubit1, qubit2)

    def add_node(self, vop: str = "YC"):

        graph = self.to_rustworkx()
        graph.add_node({"id": self.num_nodes + 1, "vop": vop})

        self.simulator = ClusterState.from_rustworkx(graph).simulator
        self.num_nodes += 1

    def remove_node(self, qubits: int | list[int]):
        graph = self.to_rustworkx()

        if isinstance(qubits, int):
            qubits = [qubits]

        graph.remove_nodes_from(qubits)

        index_map = {}
        new_graph = rx.PyGraph()
        for old_index in graph.node_indices():
            new_index = new_graph.add_node(graph[old_index])

            new_graph[new_index]["id"] = new_index
            index_map[old_index] = new_index

        for u, v in graph.edge_list():
            new_graph.add_edge(index_map[u], index_map[v], None)

        self.simulator = ClusterState.from_rustworkx(new_graph).simulator
        self.num_nodes -= len(qubits)

    @classmethod
    def from_json(cls, json_data):
        """
        Convert the graph state to a JSON-serializable format.

        Returns:
            A JSON-serializable representation of the graph state.
        """

        rx_graph = rx.parse_node_link_json(json_data)

        G = cls.from_rustworkx(rx_graph)
        return G

    def to_json(self):
        """
        Convert the graph state to a JSON-serializable format.

        Returns:
            A JSON-serializable representation of the graph state.
        """
        return rx.node_link_json(self.to_rustworkx(), node_attrs=lambda node: node)

    @property
    def stabilizers(self):
        return self.simulator.stabilizer_list()

    @property
    def adjacency_matrix(self):
        return self.simulator.adjacency_matrix()

    @property
    def adjacency_list(self):
        return self.simulator.adjacency_list()

    @property
    def vertex_operators(self):
        return self.simulator.vop_list()

    def to_rustworkx(
        self, options={"stabilizer": False, "vop": True, "neighbors": False}
    ):
        """Export data from the underlying state.

        Turning off vop = True may lead to unintended behaviour when recreating the state.

        Args:
            options (dict, optional): Default options. Defaults to {"stabilizer" : False, "vop": True}.

        Returns:
            rx.PyGraph: rustworkx representation
        """

        g = rx.PyGraph(multigraph=False)

        g.add_nodes_from(range(self.num_nodes))

        g.add_edges_from([edge for edge in self._edge_list_from_adjacency_list()])

        for node_index in g.node_indices():
            node_data = {"id": str(node_index)}
            if options["stabilizer"]:
                node_data["stabilizer"] = self.stabilizers[node_index]
            if options["vop"]:
                node_data["vop"] = self.vertex_operators[node_index]
            if options["neighbors"]:
                node_data["neighbors"] = self.adjacency_list[node_index]

            g[node_index] = node_data

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

        for node in graph.node_indices():
            if graph[node] and "vop" in graph[node]:
                vop = graphsim.LocCliffOp(graph[node]["vop"])
                c.simulator.VOP(node, vop)
        return c

    @classmethod
    def from_networkx(cls, graph: nx.Graph):
        graph = nx.convert_node_labels_to_integers(graph)
        c = cls(len(graph))
        for i in graph.nodes():
            c.H(i)

        for e in graph.edges():
            c.CZ(e[0], e[1])

        for node in graph.nodes():
            if graph.nodes[node].get("vop"):
                vop = graphsim.LocCliffOp(graph.nodes[node]["vop"])
                c.simulator.VOP(node, vop)
        return c

    def to_networkx(
        self, options={"stabilizer": False, "vop": True, "neighbors": False}
    ):
        """Export data from the underlying state.

        Turning off vop = True may lead to unintended behaviour when recreating the state.

        Args:
            options (dict, optional): Default options. Defaults to {"stabilizer" : False, "vop": True}.

        Returns:
            nx.Graph: networkx representation
        """
        g = nx.Graph()

        g.add_nodes_from(range(self.num_nodes))
        g.add_edges_from(
            [(edge[0], edge[1]) for edge in self._edge_list_from_adjacency_list()]
        )

        for node_index in g.nodes():
            node_data = {"id": str(node_index)}
            if options["stabilizer"]:
                node_data["stabilizer"] = self.stabilizers[node_index]
            if options["vop"]:
                node_data["vop"] = self.vertex_operators[node_index]
            if options["neighbors"]:
                node_data["neighbors"] = self.adjacency_list[node_index]

            g.add_node(node_index, **node_data)

        return g

    def to_cytoscape(self, export_elements=False):
        """
        Export to cytoscape.
        """

        # Subsequent calls to modify the elements in dash-cytoscape use a very specialized format
        if export_elements:
            graph = self.to_rustworkx()
            cyto_data_elements = []
            for node_index in graph.node_indices():
                cyto_data_elements.append(
                    {
                        "data": {
                            "id": str(node_index),
                            "label": str(node_index),
                            "value": node_index,
                            "vop": self.vertex_operators[node_index],
                        }
                    }
                )

            for edge_index in graph.edge_list():
                cyto_data_elements.append(
                    {
                        "data": {
                            "source": str(edge_index[0]),
                            "target": str(edge_index[1]),
                        }
                    }
                )

            return cyto_data_elements
        else:
            graph = self.to_networkx()
            cyto_data = nx.cytoscape_data(graph)

            # Processing so that labels show up properly
            for i in cyto_data["elements"]["nodes"]:
                i["data"]["label"] = i["data"].pop("name")

        return cyto_data

    @classmethod
    def from_cytoscape(cls, data):
        """
        Load a simulator from cytoscape.
        """
        graph = rx.PyGraph()

        # This is the usual case when exporting from Networkx
        for d in data["elements"]["nodes"]:
            graph.add_node({"vop": d["data"]["vop"]})

        for d in data["elements"]["edges"]:
            graph.add_edge(int(d["data"]["source"]), int(d["data"]["target"]), None)

        G = cls.from_rustworkx(graph)
        return G

    def draw(self, label_func=lambda node: str(node), **kwargs):
        g = self.to_rustworkx()
        mpl_draw(g, with_labels=True, labels=label_func, **kwargs)
