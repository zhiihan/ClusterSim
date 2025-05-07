from textwrap import dedent as d
from dash import dcc, html
import dash_bootstrap_components as dbc

move_log = dbc.Card(
    dbc.CardBody(
        [
            dcc.Markdown(
                d(
                    """
**Move Log**

Click on points in the graph.
"""
                )
            ),
            html.Div(
                [
                    dcc.Clipboard(
                        target_id="click-data",
                        style={
                            "fontSize": 20,
                        },
                    ),
                    html.Pre(
                        id="click-data",
                        style={"border": "thin lightgrey solid", "overflowX": "scroll"},
                    ),
                ],
                style={
                    "height": "400px",
                    "overflowY": "scroll",
                },
            ),
        ]
    )
)
