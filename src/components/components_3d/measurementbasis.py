from textwrap import dedent as d
from dash import dcc, callback, Input, Output
import dash_bootstrap_components as dbc

measurementbasis = dbc.Card(
    dbc.CardBody(
        [
            dcc.Markdown(
                d(
                    """
        **Select Measurement Basis**

        Click to select the type of measurement, or LC for local complementation. Click points in the graph to apply measurement.
        """
                )
            ),
            dbc.RadioItems(
                ["Z", "Y", "X", "LC"],
                value="Z",
                id="radio-items",
                className="btn-group",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-primary",
                labelCheckedClassName="active",
            ),
        ]
    )
)


@callback(
    Output("ui", "children", allow_duplicate=True),
    Input("radio-items", "value"),
    prevent_initial_call=True,
)
def update_output(value):
    return 'You have selected "{}" basis'.format(value)
