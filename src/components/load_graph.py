from textwrap import dedent as d
from dash import dcc, html

load_graph = html.Div(
    [
        dcc.Markdown(
            d(
                """
                **Load Graph State**

                Paste data to load a graph state.
                """
            )
        ),
        dcc.Input(
            id="load-graph-input",
            type="text",
            placeholder="Load Graph State",
        ),
        html.Button("Load Graph", id="load-graph-button"),
        # dcc.Store stores the intermediate value
        dcc.Store(id="browser-data"),
        dcc.Store(id="graph-data"),
        dcc.Store(id="holes-data"),
        dcc.Store(id="draw-plot"),
        html.Div(
            id="none",
            children=[],
            style={"display": "none"},
        ),
    ]
)
