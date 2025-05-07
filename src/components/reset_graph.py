from textwrap import dedent as d
from dash import dcc, html, callback, Input, Output, State
import jsonpickle
from cluster_sim.app.grid import Grid
from cluster_sim.app.holes import Holes
from cluster_sim.app.state import BrowserState
import numpy as np
import dash_bootstrap_components as dbc

reset_graph = dbc.Card(
    dbc.CardBody(
        [
            dcc.Markdown(
                d(
                    """
            **Reset Graph State.**

            Choose cube dimensions as well as a seed. If no seed, will use a random seed.
            """
                ),
                className="dbc",
            ),
            dcc.Slider(
                min=1,
                max=16,
                step=1,
                value=4,
                tooltip={
                    "placement": "bottom",
                },
                id="xmax",
                className="dash-bootstrap",
            ),
            html.Hr(),
            dcc.Slider(
                min=1,
                max=16,
                step=1,
                value=4,
                tooltip={
                    "placement": "bottom",
                },
                id="ymax",
                className="dash-bootstrap",
            ),
            html.Hr(),
            dcc.Slider(
                min=1,
                max=16,
                step=1,
                value=4,
                tooltip={
                    "placement": "bottom",
                },
                id="zmax",
                className="dash-bootstrap",
            ),
            html.Hr(),
            dbc.Stack(
                [
                    dbc.Button("Reset Grid", id="reset"),
                    dbc.Button("Undo", id="undo"),
                ],
                gap=3,
                direction="horizontal",
            ),
        ]
    )
)


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
