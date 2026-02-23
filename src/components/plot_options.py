from textwrap import dedent as d
from dash import dcc, html, callback, Input, Output, State
from cluster_sim.app import BrowserState, grid_graph_3d
from cluster_sim.simulator import ClusterState
import dash_bootstrap_components as dbc

plotoptions = dbc.Card(
    dbc.CardBody(
        [
            dbc.Checklist(
                options=[
                    {"label": "Stabilizers", "value": 0},
                    {"label": "VOP", "value": 1},
                    {"label": "Coord", "value": 2},
                ],
                value=[1, 2],
                id="display_options",
                inline=True,
                switch=True,
            ),
        ]
    )
)