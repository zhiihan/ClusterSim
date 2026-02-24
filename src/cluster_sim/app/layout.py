import rustworkx as rx
import numpy as np
from cluster_sim.app import BrowserState
from cluster_sim.simulator import ClusterState
import plotly.graph_objects as go


class Grid3D:
    def __init__(self, browser_state: BrowserState):
        self.shape = browser_state.shape
        self.browser_state = browser_state    

    def get_node_coords(self, index: int, log=False):
        """
        Get node coordinates from the grid shape and index.
        """
        index_x = index % self.shape[0]
        index_y = (index // self.shape[0]) % self.shape[1]
        index_z = (index // (self.shape[0] * self.shape[1])) % self.shape[2]

        return np.array([index_x, index_y, index_z])

    def get_node_index(self, x: int, y: int, z: int) -> int:
        """
        Get the index from the coordinates.
        """
        return x + y * self.shape[0] + z * self.shape[1] * self.shape[0]


    def update_graph_with_layout(self, g: rx.PyGraph):
        """Add the current layout to the PyGraph.

        Args:
            g (rx.PyGraph): PyGraph
        """
        for node in g.node_indices():
            g[node]["coord"] = self.get_node_coords(node)
           
        return g

layouts = {"Grid3D": Grid3D}


def update_plot_plotly(
    browser_state: BrowserState,
    G: ClusterState,
    **plot_options
):
    """
    Main function that updates the plot.
    """
    
    g = G.to_rustworkx(options={"stabilizer" : browser_state.plot_options['stabilizer'], "vop": True, 'neighbors': browser_state.plot_options['neighbors']})

    layout = layouts[browser_state.layout](
        browser_state=browser_state
    )

    g = layout.update_graph_with_layout(g)

    g_nodes, g_edges, node_hover_data = rx_graph_to_plot(g, browser_state)

    trace_edges = go.Scatter3d(
        x=g_edges[:, 0],
        y=g_edges[:, 1],
        z=g_edges[:, 2],
        mode="lines",
        line=dict(color="black", width=4),
        hoverinfo="none",
    )

    trace_nodes = go.Scatter3d(
        x=g_nodes[:, 0],
        y=g_nodes[:, 1],
        z=g_nodes[:, 2],
        mode="markers",
        marker=dict(symbol="circle", size=10, color="skyblue"),
    )

    trace_nodes.text = node_hover_data

    # Include the traces we want to plot and create a figure
    data = [trace_nodes, trace_edges]
    fig = go.Figure(data=data)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        autosize=True,
        scene_camera=browser_state.camera_state["scene.camera"],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


def rx_graph_to_plot(graph : rx.PyGraph, browser_state : BrowserState):
    """
    Convert a rustworkx object to a plotly object.
    """
    num_nodes = graph.num_nodes()

    node_coords = np.zeros((num_nodes, 3))
    node_hover_data = []

    for node_index in graph.node_indices():
        node_coords[node_index, :] = graph[node_index]["coord"]
        node_hover_data.append(_display_hover_text(graph, browser_state, node_index))

    if browser_state.plot_options['remove_isolated']:
        # FIXME: does not do anything yet
        pass 

    edge_coords = np.zeros((3 * graph.num_edges(), 3))

    for edge_index, edge in enumerate(graph.edge_list()):
        node_index_1, node_index_2 = edge

        edge_coords[3 * edge_index] = graph[node_index_1]["coord"]
        edge_coords[3 * edge_index + 1] = graph[node_index_2]["coord"]
        edge_coords[3 * edge_index + 2] = np.array(
            [np.nan, np.nan, np.nan]
        )  # Required to draw in plotly

    return node_coords, edge_coords, node_hover_data

def _display_hover_text(graph : rx.PyGraph, browser_state: BrowserState, node_index : int) -> str:
    """Used in graph_to_plot as part of update_plot_plotly.

    Args:
        graph (rx.PyGraph): rustworkx Pygraph
        browser_state (BrowserState): Browser state
        node_index (int): node index 

    Returns:
        str: text displayed on hover when in plotly
    """
    hover_text = ""

    for keys, value in graph[node_index].items():
        if browser_state.plot_options.get(keys):
            hover_text += f"{keys}: {value} \n"

    return hover_text


def update_plot_cytoscape(
    browser_state: BrowserState,
    G: ClusterState,
    **plot_options
):
    """
    Main function that updates the plot.
    """

    cyto_data = G.to_cytoscape()

    return cyto_data["elements"]