from textwrap import dedent as d
from dash import dcc, html, Input, Output, callback, no_update
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
                        target_id="move-log",
                        style={
                            "fontSize": 20,
                        },
                    ),
                    html.Pre(
                        id="move-log",
                        style={"border": "thin lightgrey solid", "overflowX": "scroll"},
                        children="H 0 1 2 3 4\nCZ 0 1 1 2 2 3 3 4 4 0\n",
                    ),
                ],
                style={
                    "height": "400px",
                    "overflowY": "scroll",
                },
            ),
            html.Button("as svg", id="btn-get-svg")
        ]
    )
)

@callback(
    Output("figure-app", "generateImage"),
    Input("btn-get-svg", "n_clicks"),
    prevent_initial_call=True)
def get_image(n_clicks, ftype='svg'):
    if n_clicks > 0:
        return {
            'type': 'svg',
            'action': 'download',
            'filename': 'cytoscape'
        }
    return no_update
