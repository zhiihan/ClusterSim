from textwrap import dedent as d
from dash import dcc, html, callback, Output, Input, State, no_update
from cluster_sim.app import BrowserState
import dash_bootstrap_components as dbc
import jsons

hover_data = dbc.Card(
    dbc.CardBody(
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
)


@callback(Output("hover-data", "children"), [Input("basic-interactions", "hoverData")])
def display_hover_data(hoverData):
    return jsons.dumps(hoverData, indent=2)


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
    if browser_data is None:
        return no_update, no_update

    s = BrowserState.from_json(browser_data)

    if relayoutData and "scene.camera" in relayoutData:
        s.camera_state = relayoutData
        return jsons.dumps(relayoutData, indent=2), s.to_json()
    else:
        return camera, s.to_json()
