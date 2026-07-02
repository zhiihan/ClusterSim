from cluster_sim.simulator import ClusterState
from dash import dcc, callback, Input, Output, State, no_update, callback_context, html
import dash_bootstrap_components as dbc
from typing import Any
import random

fusion_menu = dbc.Row(
    [
        dbc.Col(
            [
                dcc.Markdown("**Fusion Type**", style={"margin": "0 0 4px 0"}),
                dbc.Select(
                    id="fusion-type",
                    options=[
                        {"label": "XXZZ (default)", "value": "XXZZ"},
                        {"label": "XZZX", "value": "XZZX"},
                        {"label": "XYYZ", "value": "XYYZ"},
                    ],
                    value="XXZZ",
                ),
            ],
            width=3
        ),
        dbc.Col(
            [
                dcc.Markdown("**Force Measurement**", style={"margin": "0 0 4px 0"}),
                dbc.Select(
                    id="fusion-force-measurement",
                    options=[
                        {"label": "Force Qubit 1=0, Qubit 2=0", "value": 0},
                        {"label": "Force Qubit 1=1, Qubit 2=0", "value": 1},
                        {"label": "Force Qubit 1=0, Qubit 2=1", "value": 2},
                        {"label": "Force Qubit 1=1, Qubit 2=1", "value": 3},
                        {"label": "Random measurement", "value": -1},
                    ],
                    value=0,
                ),
            ],
            width=4
        ),
        dbc.Col(
            [
                dcc.Markdown("**Fusion Mode**", style={"margin": "0 0 4px 0"}),
                dbc.Select(
                    id="fusion-mode",
                    options=[
                        {"label": "Success (default)", "value": "success"},
                        {"label": "Failure", "value": "failure"},
                        {"label": "Random", "value": "random"},
                    ],
                    value="success",
                ),
            ],
            width=3
        ),
        dbc.Col(
            [
                dcc.Markdown("**Action**", style={"margin": "0 0 4px 0"}),
                dbc.Button("Fusion", outline=True, color="primary", id="fusion-gate", style={"width": "100%"}),
            ],
            width=2,
            className="d-flex flex-column justify-content-end"
        ),
    ]
)
