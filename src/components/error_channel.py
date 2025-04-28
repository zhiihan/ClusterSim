from textwrap import dedent as d
from dash import dcc, html

error_channel = html.Div(
    [
        dcc.Markdown(
            d(
                """
            **Damage the Grid.**

            Select a probability p to randomly remove nodes.
            """
            )
        ),
        dcc.Slider(
            0,
            0.3,
            step=0.03,
            value=0.06,
            tooltip={
                "placement": "bottom",
                "always_visible": True,
            },
            id="prob",
        ),
        html.Div(
            [
                html.Button("Damage Grid", id="reset-seed"),
                dcc.Input(
                    id="load-graph-seed",
                    type="number",
                    placeholder="Seed",
                ),
            ]
        ),
    ]
)
