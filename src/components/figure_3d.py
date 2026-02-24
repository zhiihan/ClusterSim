from textwrap import dedent as d
from cluster_sim.app import BrowserState, update_plot, grid_graph_3d
from cluster_sim.simulator import ClusterState
from dash import dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc


# Initialize the state of the user's browsing section
def _init_state():
    """
    Initialize the state of the user's browsing section.
    """

    browser_state = BrowserState()
    G = ClusterState.from_rustworkx(grid_graph_3d(browser_state.shape))

    return update_plot(browser_state, G)


figure_3d = dcc.Graph(
    id="basic-interactions",
    figure=_init_state(),
    responsive=True,
    style={"width": "100%", "height": "100%"},
)



@callback(
    Output("basic-interactions", "figure"),
    Input("draw-plot", "data"),
    State("basic-interactions", "relayoutData"),
    State("browser-data", "data"),
    State("graph-data", "data"),
)
def draw_plot(draw_plot, relayoutData, browser_data, graphData):
    """
    Called when ever the plot needs to be drawn.
    """
    if browser_data is None:
        return no_update

    s = BrowserState.from_json(browser_data)
    G = ClusterState.from_json(graphData)

    fig = update_plot(s, G)
    # Make sure the view/angle stays the same when updating the figure
    if "scene.camera" in relayoutData:
        fig.update_layout(scene_camera=s.camera_state["scene.camera"])
    return fig
