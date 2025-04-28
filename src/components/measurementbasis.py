from textwrap import dedent as d
from dash import dcc, html

measurementbasis = html.Div(
    [
        dcc.Markdown(
            d(
                """
        **Select Measurement Basis**

        Click to select the type of measurement. Click points in the graph to apply measurement.
        """
            )
        ),
        dcc.RadioItems(
            ["Z", "Y", "X", "Z:Hole"],
            "Z",
            id="radio-items",
            inline=True,
        ),
    ]
)
