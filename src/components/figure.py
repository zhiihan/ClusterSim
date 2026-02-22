from textwrap import dedent as d
from cluster_sim.app import BrowserState, update_plot, grid_graph_3d
from cluster_sim.simulator import ClusterState
from dash import dcc, callback, Input, Output, State, no_update
import jsonpickle
import dash_bootstrap_components as dbc

# Initialize the state of the user's browsing section
def _init_state():
    """
    Initialize the state of the user's browsing section.
    """

    s = BrowserState()

    G = ClusterState.from_rustworkx(grid_graph_3d(s.shape))
    return update_plot(s, G)


figure = dcc.Graph(
    id="basic-interactions",
    figure=_init_state(),
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
                ],
                value=["Qubits"],
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
)
def draw_plot(draw_plot, plotoptions, relayoutData, browser_data, graphData):
    """
    Called when ever the plot needs to be drawn.
    """
    if browser_data is None:
        return no_update

    s = jsonpickle.decode(browser_data)

    G = ClusterState.from_json(graphData)

    fig = update_plot(s, G, plotoptions=plotoptions)
    # Make sure the view/angle stays the same when updating the figure
    if "scene.camera" in relayoutData:
        fig.update_layout(scene_camera=s.camera_state["scene.camera"])
    return fig
