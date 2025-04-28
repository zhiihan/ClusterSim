from textwrap import dedent as d
from cluster_sim.app.grid import Grid
from cluster_sim.app.holes import Holes
from cluster_sim.app.state import BrowserState
from cluster_sim.app.utils import (
    update_plot,
)

import dash
from dash import dcc, html, callback, Input, Output, State
import jsonpickle
import numpy as np


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

display_options = html.Div(
    [
        dcc.Markdown(
            d(
                """
        **Select display options**
        """
            )
        ),
        dcc.Checklist(
            ["Qubits", "Holes", "Lattice"],
            ["Qubits", "Holes", "Lattice"],
            id="plotoptions",
        ),
    ]
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


@callback(
    Output("draw-plot", "data", allow_duplicate=True),
    Output("click-data", "children", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("reset", "n_clicks"),
    State("xmax", "value"),
    State("ymax", "value"),
    State("zmax", "value"),
    prevent_initial_call=True,
)
def reset_grid(n_clicks, xslider, yslider, zslider):
    """
    Reset the grid.
    """
    s = BrowserState()
    s.xmax = int(xslider)
    s.ymax = int(yslider)
    s.zmax = int(zslider)
    s.shape = [s.xmax, s.ymax, s.zmax]
    s.removed_nodes = np.zeros(s.xmax * s.ymax * s.zmax, dtype=bool)
    G = Grid(s.shape)
    D = Holes(s.shape)
    # Make sure the view/angle stays the same when updating the figure
    return (
        1,
        s.log,
        "Created grid of shape {}".format(s.shape),
        jsonpickle.encode(s),
        G.encode(),
        D.encode(),
    )
