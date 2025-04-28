from textwrap import dedent as d
from dash import dcc, html

reset_graph = html.Div(
    [
        dcc.Markdown(
            d(
                """
            **Reset Graph State.**

            Choose cube dimensions as well as a seed. If no seed, will use a random seed.
            """
            )
        ),
        dcc.Slider(
            1,
            15,
            step=1,
            value=5,
            tooltip={
                "placement": "bottom",
                "always_visible": True,
            },
            id="xmax",
        ),
        dcc.Slider(
            1,
            15,
            step=1,
            value=5,
            tooltip={
                "placement": "bottom",
                "always_visible": True,
            },
            id="ymax",
        ),
        dcc.Slider(
            1,
            15,
            step=1,
            value=5,
            tooltip={
                "placement": "bottom",
                "always_visible": True,
            },
            id="zmax",
        ),
        html.Button("Reset Grid", id="reset"),
        html.Button("Undo", id="undo"),
    ]
)
