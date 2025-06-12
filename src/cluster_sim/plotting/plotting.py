from cluster_sim.simulator import ClusterState, RustworkXState
import plotly.graph_objects as go
import rustworkx as rx


class Plot2D:
    def __init__(self, graph):
        self.graph = graph

    def plot(self):
        # Placeholder for plotting logic
        pass


class Plot3D:
    def __init__(self, graph):
        self.graph = graph

    def plot(self):
        # Placeholder for plotting logic
        pass


class Plot3DGrid:
    def __init__(
        self,
        cluster_state: ClusterState | rx.PyGraph,
        shape: list[int],
        browser_state=None,
    ):
        self.shape = shape
        self.browser_state = browser_state

        if isinstance(cluster_state, ClusterState) or isinstance(
            cluster_state, RustworkXState
        ):
            self.cluster_state = cluster_state
        elif isinstance(cluster_state, rx.PyGraph):
            self.cluster_state = RustworkXState(cluster_state)
        else:
            raise TypeError(
                "cluster_state must be an instance of ClusterState, RustworkXState, or rustworkx.PyGraph"
            )

    def plot(self):
        trace_nodes, trace_edges = self.rx_to_plot(index=True)
        data = [trace_nodes, trace_edges]

        fig = go.Figure(data=data)
        # fig.layout.height = 600

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            # autosize=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        if self.browser_state:
            fig.update_layout(
                scene_camera=self.browser_state.camera_state["scene.camera"]
            )

        return fig

    def _get_node_index(self, x: int, y: int, z: int):
        return x + y * self.shape[0] + z * self.shape[1] * self.shape[0]

    def _get_node_coords(self, i: int):
        index_x = i % self.shape[0]
        index_y = (i // self.shape[0]) % self.shape[1]
        index_z = (i // (self.shape[0] * self.shape[1])) % self.shape[2]
        return (index_x, index_y, index_z)

    def rx_to_plot(self, index: bool = True):
        """
        Convert a rustworkx graph to a format suitable for Plotly 3D plotting, on a Grid layout.

        Args:
            index (bool): If True, use node indices for coordinates. If False, use actual coordinates.
        """

        x_nodes, y_nodes, z_nodes = [], [], []
        x_edges, y_edges, z_edges = [], [], []

        for node in self.cluster_state.graph.node_indices():
            print(self.cluster_state.graph[node]["index"])

            x = self._get_node_coords(self.cluster_state.graph[node]["index"])

            x_nodes.append(x[0])
            y_nodes.append(x[1])
            z_nodes.append(x[2])

        for edge in self.cluster_state.graph.edge_indices():
            node1, node2 = self.cluster_state.graph.get_edge_endpoints_by_index(edge)

            x1 = self._get_node_coords(self.cluster_state.graph[node1]["index"])
            x2 = self._get_node_coords(self.cluster_state.graph[node2]["index"])

            x_edges += [x1[0], x2[0], None]
            y_edges += [x1[1], x2[1], None]
            z_edges += [x1[2], x2[2], None]

        trace_nodes = go.Scatter3d(
            x=x_nodes,
            y=y_nodes,
            z=z_nodes,
            mode="markers",
            marker=dict(symbol="circle", size=10, color="skyblue"),
        )

        trace_edges = go.Scatter3d(
            x=x_edges,
            y=y_edges,
            z=z_edges,
            mode="lines",
            line=dict(color="black", width=2),
            hoverinfo="none",
        )

        return trace_nodes, trace_edges
