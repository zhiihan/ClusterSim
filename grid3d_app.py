from cluster_sim.app import BrowserState, grid_graph_3d, layouts
from dash import html, Input, Output, State, no_update, Dash, dcc
import time
from dash_resizable_panels import PanelGroup, Panel, PanelResizeHandle
from components.components_3d import (
    figure_3d, tab_ui_3d
)
from cluster_sim.simulator import ClusterState

import dash_bootstrap_components as dbc
import logging

# Basic configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdn.jsdelivr.net/gh/zhiihan/ClusterSim/src/assets/bootstrap.min.css",
    ],
)

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
                        figure_3d,
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
                    children=tab_ui_3d,
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

    G = ClusterState.from_rustworkx(grid_graph_3d(browser_state.shape))

    return browser_state.to_json(), G.to_json()


@app.callback(
    Output("click-data", "children"),
    Output("draw-plot", "data"),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Input("figure-app", "clickData"),
    State("radio-items", "value"),
    State("click-data", "children"),
    State("figure-app", "hoverData"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    prevent_initial_call=True,
)
def measure_qubit(
    clickData, measurementChoice, clickLog, hoverData, browser_data, graphData
):
    """
    Updates the browser state if there is a click.
    """
    if not clickData:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )
    point = clickData["points"][0]

    # Do nothing if clicked on edges
    if point["curveNumber"] > 0 or "x" not in point:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )
    else:
        browser_state = BrowserState.from_json(browser_data)
        G = ClusterState.from_json(graphData)

        layout = layouts[browser_state.layout](
            browser_state=browser_state
        )

        i = layout.get_node_index(point["x"], point["y"], point["z"])

        # Click is local complementation
        if measurementChoice == "LC":
            G.local_complementation(i)
            ui = f"Applied local complementation to {layout.get_node_coords(i)}"
        if measurementChoice in ["X", "Y", "Z"] and (
            i not in browser_state.removed_nodes
        ):
            browser_state.removed_nodes.add(i)
            G.measure(i, measurementChoice)
            ui = f"Measured {layout.get_node_coords(i)} with {measurementChoice}"
        else:
            ui = "Qubit already measured!"

        browser_state.log += f"{layout.get_node_coords(i)}, {measurementChoice};\n"
        # This solves the double click issue
        time.sleep(0.1)
        return browser_state.log, i, ui, browser_state.to_json(), G.to_json()


if __name__ == "__main__":
    app.run(debug=True)
