from cluster_sim.app import BrowserState, update_plot_plotly, grid_graph_3d
from cluster_sim.simulator import ClusterState
from dash import dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
from dash import html

from components import (
    move_log,

    plotoptions,
    stabilizer,
)

from components.components_3d import (reset_graph,
    hover_data,
    zoom_data,
    load_graph,
        measurementbasis,)


# Initialize the state of the user's browsing section
def _init_state():
    """
    Initialize the state of the user's browsing section.
    """

    browser_state = BrowserState()
    G = ClusterState.from_rustworkx(grid_graph_3d(browser_state.shape))

    return update_plot_plotly(browser_state, G)


figure_3d = dcc.Graph(
    id="figure-app",
    figure=_init_state(),
    responsive=True,
    style={"width": "100%", "height": "100%"},
)


tab_1 = dbc.Col(
    [
        measurementbasis,
        # display_options
    ],
)

# tab_2 = dbc.Col(
#     [
#         algorithms,
#     ]
# )

tab_3 = dbc.Col(
    [
        reset_graph,
        load_graph,
        move_log,
    ]
)

tab_4 = dbc.Col([stabilizer])

tab_5 = dbc.Col(
    [
        plotoptions,
    ]
)

tab_6 = dbc.Col([
    hover_data,
    zoom_data,
])

tab_ui_3d = html.Div(
    [
        dbc.CardBody(
            [
                html.H2("Cluster Sim"),
                dcc.Loading(
                    dbc.Alert(
                        "Click on the graph to measure nodes.",
                        color="primary",
                        id="ui",
                    ),
                    delay_show=100,
                    delay_hide=100,
                    custom_spinner=dbc.Spinner(color="primary"),
                ),
            ]
        ),
        html.Hr(),
        dbc.Tabs(
            [
                dbc.Tab(tab_1, label="Measurements", tab_id="tab-1"),
                # dbc.Tab(tab_2, label="Algorithms", tab_id="tab-2"),
                dbc.Tab(tab_3, label="Reset and Load", tab_id="tab-3"),
                dbc.Tab(tab_4, label="Stabilizers", tab_id="tab-4"),
                dbc.Tab(tab_5, label="Plot Options", tab_id="tab-5"),
                dbc.Tab(tab_6, label="Debug", tab_id="tab-6"),
            ],
            id="tabs",
            active_tab="tab-1",
        ),
    ]
)



@callback(
    Output("figure-app", "figure"),
    Input("draw-plot", "data"),
    State("figure-app", "relayoutData"),
    State("browser-data", "data"),
    State("graph-data", "data"),
)
def draw_plot(draw_plot : int, relayoutData, browser_data, graphData):
    """
    Called when ever the plot needs to be drawn.

    draw-plot is a dummy variable. It is only to trigger the drawing of the figure after
    another component triggers it.
    """
    if browser_data is None:
        return no_update

    s = BrowserState.from_json(browser_data)
    G = ClusterState.from_json(graphData)

    fig = update_plot_plotly(s, G)
    # Make sure the view/angle stays the same when updating the figure
    if "scene.camera" in relayoutData:
        fig.update_layout(scene_camera=s.camera_state["scene.camera"])
    return fig
