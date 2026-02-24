from dash import callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
from cluster_sim.app import BrowserState

plotoptions = dbc.Card(
    dbc.CardBody(
        [
            dbc.Checklist(
                options=[
                    {"label": "Stabilizers", "value": 'stabilizer'},
                    {"label": "VOP", "value": 'vop'},
                    {"label": "Coord", "value": 'coord'},
                    {"label": "Neighbors", "value": 'neighbors'},
                    {"label": "Index", "value": 'index'},
                ],
                value=['vop', 'index', 'neighbors'],
                id="plotoptions",
                inline=True,
                switch=True,
            ),
        ]
    )
)


@callback(
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    State(component_id="browser-data", component_property="data"),
    Input(component_id='plotoptions', component_property="value"),
    prevent_initial_call=True,
)
def update_plot_options(browser_data, display_options):
    """
    Update browser data from plot options.
    """
    if browser_data is None:
        return no_update

    browser_state = BrowserState.from_json(browser_data)

    for option in browser_state.plot_options.keys():
        if option in display_options:
            browser_state.plot_options[option] = True
        else:
            browser_state.plot_options[option] = False

    return 1, "Graph loaded!", browser_state.to_json()