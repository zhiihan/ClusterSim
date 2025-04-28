from textwrap import dedent as d
from dash import dcc, html

zoom_data = html.Div(
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
