from textwrap import dedent as d
from dash import dcc, html

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
    ],
)
