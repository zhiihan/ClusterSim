from textwrap import dedent as d
from dash import dcc, html, callback, Output, Input
import json

hover_data = html.Div(
    [
        dcc.Markdown(
            d(
                """
                **Hover Data**

                Mouse over values in the graph.
            """
            )
        ),
        html.Pre(
            id="hover-data",
            style={"border": "thin lightgrey solid", "overflowX": "scroll"},
        ),
    ]
)


@callback(Output("hover-data", "children"), [Input("basic-interactions", "hoverData")])
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)
