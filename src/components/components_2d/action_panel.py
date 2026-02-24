from textwrap import dedent as d
from dash import dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

qubit_panel = dbc.Card(
    dbc.CardBody(
        [
            dcc.Markdown(
                d(
                """
                **Select Measurement Basis**

                Click points in the graph, then press the button to measure.
                """
                )
            ),
            dbc.ButtonGroup(
                [        
                    dbc.Button("MZ", outline=True, color="primary", id='MZ'),
                    dbc.Button("MY", outline=True, color="primary", id='MY'),
                    dbc.Button("MX", outline=True, color="primary", id='MX'),
                    dbc.Button("LC", outline=True, color="primary", id='LC')
                    ],
                    id='qubit-action-panel'
            )
        ]
    )
)

@callback(
    Output("ui", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Input("MZ", "n_clicks"),
    State("figure-app", "selectedNodeData"),
    prevent_initial_call=True,
)
def qubit_panel_action(n_clicks, selected_node_data):

    # TODO: modify here

    print(selected_node_data)


    return f'You have clicked MZ {n_clicks}', 1