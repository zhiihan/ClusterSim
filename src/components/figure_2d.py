import dash_cytoscape as cyto
from dash import Input, Output, State, callback, no_update
import logging
from cluster_sim.simulator import ClusterState

figure = cyto.Cytoscape(
    id="cytoscape", layout={"name": "random"}, style={"width": "100%", "height": "100%"}
)


@callback(
    Output("ui", "children", allow_duplicate=True),
    Input("cytoscape", "selectedNodeData"),
    prevent_initial_call=True,
)
def displaySelectedNodeData(data_list):
    if not data_list:
        return "Click on the graph to select nodes, or SHIFT+click to select multiple nodes."
    else:
        logging.debug(data_list)
        # return f"Selected {[i["value"] for i in data_list]}"


@callback(
    Output(
        "cytoscape",
        "elements",
    ),
    Output("circuit-info", "children"),
)
def load_simulator():
    G = ClusterState(5)
    cyto_data = G.to_cytoscape()

    return cyto_data["elements"], ""
