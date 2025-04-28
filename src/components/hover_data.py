from textwrap import dedent as d
from dash import dcc, html, callback, Output, Input, State, callback
import json
import jsonpickle

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


@callback(
    Output("relayout-data", "children"),
    Output("browser-data", "data", allow_duplicate=True),
    Input("basic-interactions", "relayoutData"),
    State("relayout-data", "children"),
    State("browser-data", "data"),
    prevent_initial_call=True,
)
def display_relayout_data(relayoutData, camera, browser_data):
    """
    Updates zoom and camera information.
    """
    if browser_data is not None:
        s = jsonpickle.decode(browser_data)

    if relayoutData and "scene.camera" in relayoutData:
        s.camera_state = relayoutData
        return json.dumps(relayoutData, indent=2), jsonpickle.encode(s)
    else:
        return camera, jsonpickle.encode(s)
