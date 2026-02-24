from cluster_sim import ClusterState
from cluster_sim.app import BrowserState, layouts
from textwrap import dedent as d
from dash import dcc, html, callback, Input, Output, State, no_update
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import itertools
import dash_bootstrap_components as dbc
import logging


algorithms = dbc.Card(
    dbc.CardBody(
        [
            dcc.Markdown(
                d(
                    """
                    **Algorithms**

                    Click on points in the graph.
                """
                )
            ),
            dbc.Stack(
                [
                    dbc.Button("RHG Lattice", id="alg1"),
                    dbc.Button("Reduction", id="reduction"),
                ],
                direction="horizontal",
                gap=3,
            ),
            html.Hr(),
            dcc.Dropdown(
                ["Select One Cube", "Select All Cubes", "Select All Connected Cubes"],
                "Select All Cubes",
                id="select-cubes",
                className="dash-bootstrap",
            ),
        ],
    )
)


@callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Input("alg1", "n_clicks"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    prevent_initial_call=True,
)
def rhg_lattice_scale(nclicks, browser_data, graphData):
    """
    Create a RHG lattice from a square lattice, with a scale factor.

    Parameters:
    - nclicks: number of clicks on the button (unused)
    - scale_factor: scale factor for the RHG lattice
    - browser_data: data from the browser
    - graphData: data from the graph
    - holeData: data from the holes
    """
    browser_state = BrowserState.from_json(browser_data)
    G = ClusterState.from_json(graphData)

    layout = layouts[browser_state.layout](
        browser_state=browser_state
    )

    for node_index in range(len(G)):
        coords = layout.get_node_coords(node_index)
        if (coords[0] % 2) == (coords[1] % 2) == (coords[2] % 2):
            G.measure(node_index, force=0, basis='Z')
            browser_state.log += f"{layout.get_node_coords(node_index)}, Z;\n"
            browser_state.removed_nodes.add(node_index)

    ui = "Created an RHG lattice."

    return browser_state.log, 1, ui, browser_state.to_json(), G.to_json()


# @callback(
#     Output("click-data", "children", allow_duplicate=True),
#     Output("draw-plot", "data", allow_duplicate=True),
#     Output("ui", "children", allow_duplicate=True),
#     Output("browser-data", "data", allow_duplicate=True),
#     Output("graph-data", "data", allow_duplicate=True),
#     Output("holes-data", "data", allow_duplicate=True),
#     Input("reduction", "n_clicks"),
#     State("browser-data", "data"),
#     State("graph-data", "data"),
#     State("holes-data", "data"),
#     State("select-cubes", "value"),
#     prevent_initial_call=True,
# )
# def reduce_lattice(
#     nclicks, browser_data, graphData, holeData, select_cubes, algorithm="line"
# ):

#     s = jsonpickle.decode(browser_data)
#     G = Grid(s.shape, json_data=graphData)
#     D = Holes(s.shape, json_data=holeData)

#     if getattr(s, "scale_factor", None) is None:
#         ui = "Find Cluster: Run RHG Lattice first."
#         return no_update, no_update, ui, no_update, no_update, no_update

#     s.valid_unit_cells, s.unit_cell_shape = generate_unit_cell_global_coords(
#         s.shape, s.scale_factor, s.offset
#     )

#     if not s.valid_unit_cells:
#         return (
#             no_update,
#             no_update,
#             "No valid unit cells found.",
#             jsonpickle.encode(s),
#             G.encode(),
#             D.encode(),
#         )

#     click_number = nclicks % (len(s.valid_unit_cells))

#     if select_cubes == "Select One Cube":
#         H, imperfection_score = find_rings(
#             G,
#             s.scale_factor,
#             unit_cell_coord=s.valid_unit_cells[click_number],
#         )
#         ui = f"Reduction: Imperfection score = {imperfection_score}"

#     elif select_cubes == "Select All Cubes":
#         graphs = []
#         imperfection_score = 0
#         for unit_cell_coord in s.valid_unit_cells:
#             H, _ = find_rings(
#                 G,
#                 s.scale_factor,
#                 unit_cell_coord=unit_cell_coord,
#             )
#             graphs.append(H)
#         H = nx.compose_all(graphs)

#         imperfection_score = nx.number_of_isolates(H)

#         logging.info(f"Total Imperfection score: {imperfection_score}")

#         ui = f"Reduction: Total Imperfection score = {imperfection_score}, Equivalent 1-cell deletion ratio = {100 * imperfection_score / H.number_of_nodes()}%"
#     elif select_cubes == "Select All Connected Cubes":

#         C = nx.Graph()
#         graphs_hashmap = {}
#         imperfect_cells = 0

#         for unit_cell_coord in s.valid_unit_cells:
#             H_subgraph, imperfection_score = find_rings(
#                 G, s.scale_factor, unit_cell_coord=unit_cell_coord
#             )
#             if imperfection_score == 0:
#                 C.add_node(tuple(unit_cell_coord))
#                 graphs_hashmap[tuple(unit_cell_coord)] = H_subgraph

#                 for c in C.nodes:
#                     if taxicab_metric(c, unit_cell_coord) <= (s.scale_factor + 1):
#                         C.add_edge(c, tuple(unit_cell_coord))
#             else:
#                 imperfect_cells += 1

#         connected_clusters = [C.subgraph(c).copy() for c in nx.connected_components(C)]

#         if not connected_clusters:
#             ui = "Reduction: No connected clusters found."
#             s.lattice = None
#             s.lattice_edges = None
#             return s.log, 1, ui, jsonpickle.encode(s), G.encode(), D.encode()

#         connected_cluster_graph = connected_clusters[nclicks % len(connected_clusters)]

#         graphs = [
#             graphs_hashmap[unit_cell_coord]
#             for unit_cell_coord in connected_cluster_graph.nodes
#         ]

#         H = nx.compose_all(graphs)

#         imperfection_score = nx.number_of_isolates(H)
#         perfect_cells = len(graphs)

#         logging.info(
#             f"Finding connected components: {perfect_cells} perfect cells, {imperfect_cells} imperfect cells."
#         )

#         ui = f"Reduction: Displaying cluster {nclicks % len(connected_clusters) + 1}/{len(connected_clusters)}, perfect cells = {perfect_cells}, imperfect cells = {imperfect_cells}"

#     nodes, edges = nx_to_plot(H, shape=s.shape, index=False)

#     lattice = go.Scatter3d(
#         x=nodes[0],
#         y=nodes[1],
#         z=nodes[2],
#         mode="markers",
#         line=dict(color="blue", width=2),
#         hoverinfo="none",
#     )

#     lattice_edges = go.Scatter3d(
#         x=edges[0],
#         y=edges[1],
#         z=edges[2],
#         mode="lines",
#         line=dict(color="blue", width=2),
#         hoverinfo="none",
#     )

#     s.lattice = lattice.to_json()
#     s.lattice_edges = lattice_edges.to_json()

#     return s.log, 1, ui, jsonpickle.encode(s), G.encode(), D.encode()

