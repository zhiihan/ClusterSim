from cluster_sim.app.state import BrowserState
from cluster_sim.app.utils import (
    get_node_index,
    get_node_coords,
)

from cluster_sim.simulator import ClusterState, NetworkXState
import rustworkx as rx
import dash
from dash import html, Input, Output, State
import time
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
    user_state = BrowserState()

    cluster = ClusterState(rx.grid_graph(user_state.shape))
    errors = NetworkXState(rx.PyGraph())

    return user_state.to_json(), cluster.to_json(), errors.to_json()


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
def handle_qubit_measurements(
    clickData,
    measurement_basis,
    clickLog,
    hoverData,
    user_state_json: dict,
    cluster_json: dict,
    erasure_json: dict,
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

    user_state = BrowserState.from_json(user_state_json)
    cluster = ClusterState.from_json(cluster_json)
    errors = NetworkXState.from_json(erasure_json)

    i = get_node_index(point["x"], point["y"], point["z"], user_state.shape)
    # Update the plot based on the node clicked
    if measurement_basis == "Erasure":
        errors.add_node(i)
        measurement_basis = "Z"  # Handle it as if it was Z measurement

    if not user_state.removed_nodes[i]:
        user_state.removed_nodes[i] = True

        cluster.measure(i, measurement_basis)
        user_state.move_list.append(
            [get_node_coords(i, user_state.shape), measurement_basis]
        )
        ui = f"Measured {get_node_coords(i, user_state.shape)} with {measurement_basis}"
    user_state.log.append(
        f"{get_node_coords(i, user_state.shape)}, {measurement_basis}; "
    )
    user_state.log.append(html.Br())

    # This solves the double click issue
    time.sleep(0.1)
    return (
        html.P(user_state.log),
        i,
        ui,
        user_state.to_json(),
        cluster.to_json(),
        errors.to_json(),
    )


if __name__ == "__main__":

    app.run(debug=True)
