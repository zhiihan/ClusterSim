from textwrap import dedent as d
from dash import dcc, html, callback, Input, Output, State
from cluster_sim.app import BrowserState
from cluster_sim.simulator import ClusterState, NetworkXState

import dash_bootstrap_components as dbc

settings = dbc.Card(
    dbc.CardBody(
        [
            dcc.Markdown(
                d(
                    """
            **Plot settings**

            Adjust plot colors.
            """
                ),
                className="dbc",
            ),
            dbc.Input(
                type="color",
                id="colorpicker",
                value="#000000",
                style={"width": 75, "height": 50},
            ),
        ]
    )
)
