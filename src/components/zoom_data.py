from textwrap import dedent as d
from dash import dcc, html
import dash_bootstrap_components as dbc

zoom_data = dbc.Card(
    dbc.CardBody(
        [
            dcc.Markdown(
                d(
                    """
                **Zoom and Relayout Data**

                Click and drag on the graph to zoom or click on the zoom
                buttons in the graph's menu bar.
                Clicking on legend items will also fire
                this event.
            """
                )
            ),
            html.Pre(
                id="relayout-data",
                style={"border": "thin lightgrey solid", "overflowX": "scroll"},
            ),
        ]
    )
)
