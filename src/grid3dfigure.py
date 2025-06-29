from cluster_sim.app.grid import Grid
from cluster_sim.app.holes import Holes
from cluster_sim.app.state import BrowserState
from cluster_sim.app.utils import (
    get_node_index,
    get_node_coords,
)
import dash
from dash import html, Input, Output, State
import time
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from dash_resizable_panels import PanelGroup, Panel, PanelResizeHandle
from components import (
    figure,
    tab_ui,
)
import dash_bootstrap_components as dbc
import logging

# Basic configuration
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

jsonpickle_numpy.register_handlers()

app = dash.Dash(
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
    Output("holes-data", "data"),
    Input("none", "children"),
)
def initial_call(dummy):
    """
    Initialize the graph in the browser as a JSON object.
    """
    s = BrowserState()
    G = Grid(s.shape)
    D = Holes(s.shape)

    return jsonpickle.encode(s), G.encode(), D.encode()


@app.callback(
    Output("click-data", "children"),
    Output("draw-plot", "data"),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("basic-interactions", "clickData"),
    State("radio-items", "value"),
    State("click-data", "children"),
    State("basic-interactions", "hoverData"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("holes-data", "data"),
    prevent_initial_call=True,
)
def display_click_data(
    clickData, measurementChoice, clickLog, hoverData, browser_data, graphData, holeData
):
    """
    Updates the browser state if there is a click.
    """
    if not clickData:
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )
    point = clickData["points"][0]

    # Do nothing if clicked on edges
    if point["curveNumber"] > 0 or "x" not in point:
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )
    else:
        s = jsonpickle.decode(browser_data)
        G = Grid(s.shape, json_data=graphData)
        D = Holes(s.shape, json_data=holeData)

        i = get_node_index(point["x"], point["y"], point["z"], s.shape)
        # Update the plot based on the node clicked
        if measurementChoice == "Erasure":
            D.add_node(i)
            measurementChoice = "Z"  # Handle it as if it was Z measurement
        if not s.removed_nodes[i]:
            s.removed_nodes[i] = True
            G.handle_measurements(i, measurementChoice)
            s.move_list.append([get_node_coords(i, s.shape), measurementChoice])
            ui = f"Measured {get_node_coords(i, s.shape)} with {measurementChoice}"
        s.log.append(f"{get_node_coords(i, s.shape)}, {measurementChoice}; ")
        s.log.append(html.Br())

        # This solves the double click issue
        time.sleep(0.1)
        return html.P(s.log), i, ui, jsonpickle.encode(s), G.encode(), D.encode()


if __name__ == "__main__":

    app.run(debug=True)
