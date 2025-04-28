from textwrap import dedent as d
from dash import dcc, html, callback, Input, Output
from dash_bootstrap_components import RadioItems

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
            ["Z", "Y", "X", "Erasure"],
            "Z",
            id="radio-items",
            inline=True,
        ),
    ]
)


@callback(
    Output("ui", "children", allow_duplicate=True),
    Input("radio-items", "value"),
    prevent_initial_call=True,
)
def update_output(value):
    return 'You have selected "{}" basis'.format(value)
