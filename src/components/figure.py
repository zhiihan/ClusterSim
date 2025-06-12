from textwrap import dedent as d
from cluster_sim.simulator import ClusterState, RustworkXState
from cluster_sim.app.state import BrowserState

import dash
from dash import dcc, callback, Input, Output, State
from cluster_sim.plotting import Plot3DGrid
from networkx import grid_graph
from plotly.io import from_json
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


# Initialize the state of the user's browsing section
def _initial_figure():
    """
    Initialize the figure with a 3D grid graph.
    """

    user_state = BrowserState()

    G = ClusterState(grid_graph(dim=user_state.shape))
    return Plot3DGrid(G, user_state.shape).plot()


figure = dcc.Graph(
    id="basic-interactions",
    figure=_initial_figure(),
    responsive=True,
    style={"width": "100%", "height": "100%"},
)

display_options = dbc.Card(
    dbc.CardBody(
        [
            dcc.Markdown(
                d(
                    """
        **Display Options**

        Select what to display in the graph.
        """
                )
            ),
            dbc.Checklist(
                options=[
                    {"label": "Qubits", "value": "Qubits"},
                    {"label": "Erasures", "value": "Holes"},
                    {"label": "Lattice", "value": "Lattice"},
                ],
                value=["Qubits", "Holes", "Lattice"],
                id="plotoptions",
                inline=True,
                switch=True,
            ),
        ]
    )
)


@callback(
    Output("basic-interactions", "figure"),
    Input("draw-plot", "data"),
    Input("plotoptions", "value"),
    State("basic-interactions", "relayoutData"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("holes-data", "data"),
)
def draw_plot(draw_plot, plotoptions, relayoutData, browser_data, graphData, holeData):
    """
    Called when ever the plot needs to be drawn.
    """
    if browser_data is None:
        return dash.no_update

    user_state = BrowserState.from_json(browser_data)
    G = ClusterState.from_json(graphData)
    D = RustworkXState.from_json(holeData)
    trace_nodes, trace_edges = Plot3DGrid(G, user_state.shape).rx_to_plot()
    trace_holes, trace_holes_edges = Plot3DGrid(D, user_state.shape).rx_to_plot()

    if "Qubits" in plotoptions:
        trace_nodes.visible = True
        trace_edges.visible = True
    else:
        trace_nodes.visible = "legendonly"
        trace_edges.visible = "legendonly"

    if "Holes" in plotoptions:
        trace_holes.visible = True
        trace_holes_edges.visible = True
    else:
        trace_holes.visible = "legendonly"
        trace_holes_edges.visible = "legendonly"

    # Include the traces we want to plot and create a figure
    data = [trace_nodes, trace_edges, trace_holes, trace_holes_edges]
    if user_state.lattice:
        lattice_nodes = go.Scatter3d(
            from_json(user_state.lattice_edges).data[0],
            mode="markers",
            line=dict(color="blue", width=2),
            hoverinfo="none",
        )
        if "Lattice" in plotoptions:
            lattice_nodes.visible = True
        else:
            lattice_nodes.visible = "legendonly"
        data.append(lattice_nodes)
    if user_state.lattice_edges:
        lattice_edges = go.Scatter3d(
            from_json(user_state.lattice_edges).data[0],
            mode="lines",
            line=dict(color="blue", width=2),
            hoverinfo="none",
        )
        if "Lattice" in plotoptions:
            lattice_edges.visible = True
        else:
            lattice_edges.visible = "legendonly"
        data.append(lattice_edges)

    fig = go.Figure(data=data)
    # fig.layout.height = 600
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        # autosize=True,
        scene_camera=user_state.camera_state["scene.camera"],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    if "scene.camera" in relayoutData:
        fig.update_layout(scene_camera=user_state.camera_state["scene.camera"])
    return fig
