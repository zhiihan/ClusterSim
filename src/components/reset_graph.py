from textwrap import dedent as d
from dash import dcc, html, callback, Input, Output, State
from cluster_sim.app import BrowserState, grid_graph_3d
from cluster_sim.simulator import ClusterState
import dash_bootstrap_components as dbc
import numpy as np

reset_graph = dbc.Card(
    dbc.CardBody(
        [
            dcc.Markdown(
                d(
                    """
            **Reset Graph State**

            Choose cube dimensions.
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
    browser_state = BrowserState()
    browser_state.xmax = int(xslider)
    browser_state.ymax = int(yslider)
    browser_state.zmax = int(zslider)
    browser_state.shape = (browser_state.xmax, browser_state.ymax, browser_state.zmax)
    browser_state.removed_nodes = set()

    G = ClusterState.from_rustworkx(grid_graph_3d(browser_state.shape))

    # Make sure the view/angle stays the same when updating the figure
    return (
        1,
        browser_state.log,
        "Created grid of shape {}".format(browser_state.shape),
        browser_state.to_json(),
        G.to_json(),
    )
