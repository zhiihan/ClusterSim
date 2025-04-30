from textwrap import dedent as d
from dash import dcc, html, callback, Input, Output, State
import jsonpickle
import numpy as np
from cluster_sim.app.utils import get_node_index
from cluster_sim.app.grid import Grid
from cluster_sim.app.holes import Holes
import itertools

algorithms = html.Div(
    [
        dcc.Markdown(
            d(
                """
                    **Algorithms**

                    Click on points in the graph.
                """
            )
        ),
        html.Button("RHG Lattice", id="alg1"),
        html.Button("Find Lattice", id="findlattice"),
        html.Button("Find Cluster", id="alg2"),
        html.Button("Repair Grid", id="repair"),
        html.Button("Find Percolation", id="alg3"),
        html.Button("RHG Rules", id="RHGrules"),
        dcc.Slider(
            id="rhg-slider",
            min=2,
            max=4,
            step=1,
            value=3,
            tooltip={
                "placement": "bottom",
                "always_visible": True,
            },
        ),
    ],
)


# @callback(
#     Output("click-data", "children", allow_duplicate=True),
#     Output("draw-plot", "data", allow_duplicate=True),
#     Output("ui", "children", allow_duplicate=True),
#     Output("browser-data", "data", allow_duplicate=True),
#     Output("graph-data", "data", allow_duplicate=True),
#     Output("holes-data", "data", allow_duplicate=True),
#     Input("RHGrules", "n_clicks"),
#     State("browser-data", "data"),
#     State("graph-data", "data"),
#     State("holes-data", "data"),
#     prevent_initial_call=True,
# )
# def rhgrules_extended(nclicks, browser_data, graphData, holeData):
#     """
#     Create a RHG lattice from a square lattice.

#     Somehow this works for n = 5.
#     """
#     s = jsonpickle.decode(browser_data)
#     G = Grid(s.shape, json_data=graphData)
#     D = Holes(s.shape, json_data=holeData)

#     holes = D.graph.nodes
#     hole_locations = np.zeros(27)
#     xoffset, yoffset, zoffset = s.offset

#     # counting where the holes are
#     for h in holes:
#         x, y, z = h

#         x_vec = np.array([x, y, z])
#         for xoffset, yoffset, zoffset in itertools.product(range(3), repeat=3):
#             test_cond = (x_vec + np.array([xoffset, yoffset, zoffset])) % 4
#             if np.all(test_cond == 0) or np.all(test_cond != 0):
#                 hole_locations[xoffset + yoffset * 3 + zoffset * 9] += 1

#     xoffset = np.argmax(hole_locations) % 3
#     yoffset = np.argmax(hole_locations) // 3
#     zoffset = np.argmax(hole_locations) // 9

#     s.offset = [xoffset, yoffset, zoffset]

#     print(hole_locations)

#     for z in range(s.shape[2]):
#         for y in range(s.shape[1]):
#             for x in range(s.shape[0]):
#                 x_vec = (
#                     np.array([x, y, z]) + np.array([xoffset, yoffset, zoffset])
#                 ) % 4

#                 if np.all(x_vec == 0) or np.all(x_vec != 0):
#                     i = get_node_index(x, y, z, s.shape)
#                     if s.removed_nodes[i] == False:
#                         G.handle_measurements(i, "Z")
#                         s.log.append(f"{i}, Z; ")
#                         s.log.append(html.Br())
#                         s.removed_nodes[i] = True
#                         s.move_list.append([i, "Z"])

#     s.cubes, s.n_cubes = D.find_lattice(s.removed_nodes, xoffset, yoffset, zoffset)
#     ui = f"RHG rules"

#     return s.log, 1, ui, jsonpickle.encode(s), G.encode(), D.encode()


@callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("RHGrules", "n_clicks"),
    State("rhg-slider", "value"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("holes-data", "data"),
    prevent_initial_call=True,
)
def rhgrules_extended(nclicks, scale_factor, browser_data, graphData, holeData):
    """
    Create a RHG lattice from a square lattice.

    """
    s = jsonpickle.decode(browser_data)
    G = Grid(s.shape, json_data=graphData)
    D = Holes(s.shape, json_data=holeData)

    holes = D.graph.nodes
    hole_locations = np.zeros(27)
    xoffset, yoffset, zoffset = s.offset

    # counting where the holes are
    for h in holes:
        x, y, z = h

        x_vec = np.array([x, y, z])
        for xoffset, yoffset, zoffset in itertools.product(
            range(scale_factor), repeat=3
        ):
            test_cond = (x_vec + np.array([xoffset, yoffset, zoffset])) % scale_factor
            if np.all(test_cond == 0) or np.all(test_cond != 0):
                hole_locations[
                    xoffset + yoffset * scale_factor + zoffset * scale_factor**2
                ] += 1

    xoffset = np.argmax(hole_locations) % scale_factor
    yoffset = np.argmax(hole_locations) // scale_factor
    zoffset = np.argmax(hole_locations) // scale_factor**2

    s.offset = [xoffset, yoffset, zoffset]

    print(hole_locations)

    for z in range(s.shape[2]):
        for y in range(s.shape[1]):
            for x in range(s.shape[0]):
                x_vec = (
                    np.array([x, y, z]) + np.array([xoffset, yoffset, zoffset])
                ) % scale_factor

                if np.all(x_vec == 0) or np.all(x_vec != 0):
                    i = get_node_index(x, y, z, s.shape)
                    if s.removed_nodes[i] == False:
                        G.handle_measurements(i, "Z")
                        s.log.append(f"{i}, Z; ")
                        s.log.append(html.Br())
                        s.removed_nodes[i] = True
                        s.move_list.append([i, "Z"])

    s.cubes, s.n_cubes = D.find_lattice(s.removed_nodes, xoffset, yoffset, zoffset)
    ui = f"RHG rules"

    return s.log, 1, ui, jsonpickle.encode(s), G.encode(), D.encode()
