from cluster_sim.app import BrowserState, get_node_index, get_node_coords, grid_graph_3d
from dash import html, Input, Output, State, no_update, Dash
import time
import jsonpickle
from dash_resizable_panels import PanelGroup, Panel, PanelResizeHandle
from components import (
    figure,
    tab_ui,
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
                        figure,
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
                    children=tab_ui,
                    style={"overflowY": "scroll"},
                ),
            ],
            direction="horizontal",
            style={"height": "100vh"},
        )
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
    s = BrowserState()

    G = ClusterState.from_rustworkx(grid_graph_3d(s.shape))

    return s.to_json(), G.to_json()


@app.callback(
    Output("click-data", "children"),
    Output("draw-plot", "data"),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Input("basic-interactions", "clickData"),
    State("radio-items", "value"),
    State("click-data", "children"),
    State("basic-interactions", "hoverData"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    prevent_initial_call=True,
)
def display_click_data(
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
        print(browser_data)
        s = jsonpickle.decode(browser_data)
        G = ClusterState.from_json(graphData)
        i = get_node_index(point["x"], point["y"], point["z"], s.shape)

        # Click is local complementation
        if measurementChoice == "LC":
            G.LC(i)
            ui = f"Applied local complementation to {get_node_coords(i, s.shape)}"

        if measurementChoice in ["X", "Y", "Z"] and not s.removed_nodes[i]:
            s.removed_nodes[i] = True
            G.measure(i, measurementChoice)
            ui = f"Measured {get_node_coords(i, s.shape)} with {measurementChoice}"

        s.move_list.append([get_node_coords(i, s.shape), measurementChoice])
        s.log.append(f"{get_node_coords(i, s.shape)}, {measurementChoice}; ")
        s.log.append(html.Br())

        # This solves the double click issue
        time.sleep(0.1)
        return html.P(s.log), i, ui, s.to_json(), G.to_json()


if __name__ == "__main__":
    app.run(debug=True)
