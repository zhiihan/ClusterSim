from textwrap import dedent as d
from cluster_sim.app.grid import Grid
from cluster_sim.app.holes import Holes
from cluster_sim.app.state import BrowserState
from cluster_sim.app.utils import (
    update_plot,
)

import dash
from dash import dcc, callback, Input, Output, State
import jsonpickle
import dash_bootstrap_components as dbc


# Initialize the state of the user's browsing section
def _init_state():
    """
    Initialize the state of the user's browsing section.
    """

    s = BrowserState()
    G = Grid(s.shape)
    D = Holes(s.shape)
    return update_plot(s, G, D)


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

    s = jsonpickle.decode(browser_data)
    G = Grid(s.shape, json_data=graphData)
    D = Holes(s.shape, json_data=holeData)

    fig = update_plot(s, G, D, plotoptions=plotoptions)
    # Make sure the view/angle stays the same when updating the figure
    if "scene.camera" in relayoutData:
        fig.update_layout(scene_camera=s.camera_state["scene.camera"])
    return fig
