from textwrap import dedent as d
from dash import dcc, html

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
