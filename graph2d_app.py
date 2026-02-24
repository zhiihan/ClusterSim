from dash import Dash, html, dcc, Input, Output

from components.components_2d import tab_ui_2d, figure_2d
from cluster_sim.app import BrowserState
from cluster_sim.simulator import ClusterState

import dash_bootstrap_components as dbc
from dash_resizable_panels import PanelGroup, Panel, PanelResizeHandle

import logging

logging.basicConfig(level=logging.DEBUG)

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    [
        PanelGroup(
            id="main_app",
            children=[
                Panel(
                    id="resize_figure",
                    children=[
                        figure_2d,
                    ],
                ),
                PanelResizeHandle(
                    html.Div(
                        style={
                            "backgroundColor": "grey",
                            "height": "100%",
                            "width": "5px",
                        }
                    )
                ),
                Panel(
                    id="resize_info",
                    children=tab_ui_2d,
                    style={"overflowY": "scroll"},
                ),
            ],
            direction="horizontal",
            style={"height": "100vh"},
        ),
        dcc.Store(id="browser-data"),
        dcc.Store(id="graph-data"),
        dcc.Store(id="draw-plot"), # This is a dummy variable
        html.Div(
            id="none",
            children=[],
            style={"display": "none"},
        ),
    ]
)


@app.callback(
    Output("browser-data", "data"),
    Output("graph-data", "data"),
    Input("none", "children"),
)
def initial_call(dummy):
    """
    Initialize the graph in the browser as a JSON object.
    """
    browser_state = BrowserState()

    G = ClusterState(5)

    return browser_state.to_json(), G.to_json()

if __name__ == "__main__":
    app.run(debug=True)
