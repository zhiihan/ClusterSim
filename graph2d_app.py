from dash import Dash, html, dcc, Input, Output

from components.components_2d import tab_ui_2d, figure_2d
from cluster_sim.app import BrowserState
from cluster_sim.simulator import ClusterState

import dash_bootstrap_components as dbc
from dash_resizable_panels import PanelGroup, Panel, PanelResizeHandle

import logging

logging.basicConfig(level=logging.DEBUG)

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = "Cluster Sim"

server = app.server  # For deployment

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
        dcc.Store(id="draw-plot"),  # This is a dummy variable
        dcc.Store(id="clipboard-store"),
        html.Button(id="ctrl-c-btn", style={"display": "none"}),
        html.Button(id="ctrl-v-btn", style={"display": "none"}),
        html.Button(id="ctrl-z-btn", style={"display": "none"}),
        html.Button(id="backspace-btn", style={"display": "none"}),
        html.Div(
            id="none",
            children=[],
            style={"display": "none"},
        ),
        html.Div(
            id="clientside-dummy",
            children=[],
            style={"display": "none"},
        ),
    ]
)

app.clientside_callback(
    """
    function(dummy) {
        document.addEventListener('keydown', function(e) {
            // Do not capture keys if the user is typing in an input or textarea
            const active = document.activeElement;
            if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.isContentEditable)) {
                return;
            }

            // Check if Ctrl+C
            if (e.ctrlKey && (e.key === 'c' || e.key === 'C')) {
                const btn = document.getElementById('ctrl-c-btn');
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            }
            // Check if Ctrl+V
            if (e.ctrlKey && (e.key === 'v' || e.key === 'V')) {
                const btn = document.getElementById('ctrl-v-btn');
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            }
            // Check if Ctrl+Z
            if (e.ctrlKey && (e.key === 'z' || e.key === 'Z')) {
                const btn = document.getElementById('ctrl-z-btn');
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            }
            // Check if Backspace
            if (e.key === 'Backspace') {
                const btn = document.getElementById('backspace-btn');
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            }
        }, true);
        return window.dash_clientside.no_update;
    }
    """,
    Output("clientside-dummy", "children"),
    Input("none", "children"),
    prevent_initial_call=False,
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
