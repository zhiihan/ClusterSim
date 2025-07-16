import networkx as nx
from cluster_sim.app.utils import get_node_coords, taxicab_metric
import numpy as np


class Holes:
    """
    Create a Grid object as a NetworkX representation.
    """

    def __init__(self, shape, json_data=None):

        self.shape = shape

        if json_data:
            self.graph = nx.node_link_graph(json_data, edges="edges")
        elif shape:
            self.graph = nx.Graph()
        else:
            raise NotImplementedError

        self.add_edges()

    def add_node(self, i):
        self.graph.add_node(tuple(get_node_coords(i, shape=self.shape)))

    def add_edges(self):
        nodes = list(self.graph.nodes)
        for index, n in enumerate(nodes):
            for n2 in nodes[index:]:
                if taxicab_metric(np.array(n), np.array(n2)) == 1:
                    self.graph.add_edge(n, n2)

    def to_networkx(self):
        return self.graph

    # def double_hole(self):
    #     """
    #     Check if a hole is a double hole.

    #     Input: holes object
    #     Output:
    #     """

    #     for i in self.graph.nodes():
    #         for j in self.graph.nodes():
    #             x_diff = np.abs(np.array(i) - np.array(j))
    #             if np.sum(x_diff) == 2:
    #                 if not ((x_diff[0] == 2) or (x_diff[1] == 2) or (x_diff[2] == 2)):
    #                     self.graph.add_edge(i, j)

    def encode(self):
        return nx.node_link_data(self.graph, edges="edges")
