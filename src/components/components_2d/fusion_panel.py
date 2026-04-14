from cluster_sim.simulator import ClusterState
from dash import dcc, callback, Input, Output, State, no_update, callback_context, html
import dash_bootstrap_components as dbc
from typing import Any
import random

fusion_menu = dbc.Card(
        dbc.CardBody(
            [
                dcc.Markdown("**Fusion Panel**"),
                dbc.Select(
                    id="fusion-type",
                    options=[
                        {"label": "XXZZ (default)", "value": "II"},
                        {"label": "XZZX", "value": "HI"},
                    ],
                    placeholder="XXZZ (default)",
                    value="II",
                    style={"minwidth": "200px", "width": "200px"},
                ),
                dbc.Select(
                    id="fusion-force-measurement",
                    options=[
                        {"label": "Force Qubit 1 = 0, Qubit 2 = 0 (default)", "value": 0},
                        {"label": "Force Qubit 1 = 1, Qubit 2 = 0", "value": 1},
                        {"label": "Force Qubit 1 = 0, Qubit 2 = 1", "value": 2},
                        {"label": "Force Qubit 1 = 1, Qubit 2 = 1", "value": 3},
                        {"label": "Random measurement", "value": -1},
                    ],
                    placeholder="Force Qubit 1 = 0, Qubit 2 = 0 (default)",
                    value=0,
                    style={"minwidth": "200px", "width": "200px"},
                ),
                dbc.Select(
                    id="fusion-mode",
                    options=[
                        {"label": "Success (default)", "value": "success"},
                        {"label": "Failure", "value": "failure"},
                        {"label": "Random", "value": "random"},
                    ],
                    placeholder="Success (default)",
                    value=0,
                    style={"minwidth": "200px", "width": "200px"},
                ),
                dbc.Button(
                    "Fusion", outline=True, color="primary", id="fusion-gate"
                ),
            ]
        )
    )

