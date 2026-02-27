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

                        Click on points in the graph. Can be copied to clipboard to load a graph state.
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
                        children="H 0 1 2 3 4\nCZ 0 1 1 2 2 3 3 4 4 0\n",
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
