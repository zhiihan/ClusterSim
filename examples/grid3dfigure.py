import plotly.graph_objects as go
from grid import Grid
from holes import Holes 
import json
from textwrap import dedent as d
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import time
import random
import numpy as np
import networkx as nx
from helperfunctions import *

# Global constants
xmax = 7
ymax = 7
zmax = 7
shape = [xmax, ymax, zmax]
p = 0.5
seed = None
path_clicks = 0

G = Grid([xmax, ymax, zmax]) # qubits
D = Holes([xmax, ymax, zmax]) # holes
cubes = None
lattice = None
lattice_edges = None
connected_cubes = None
removed_nodes = np.zeros(xmax*ymax*zmax, dtype=bool)
log = [] #html version of move_list
move_list = [] #local variable containing moves
camera_state = {
  "scene.camera": {
    "up": {
      "x": 0,
      "y": 0,
      "z": 1
    },
    "center": {
      "x": 0,
      "y": 0,
      "z": 0
    },
    "eye": {
      "x": 1.8999654712209553,
      "y": 1.8999654712209548,
      "z": 1.8999654712209553
    },
    "projection": {
      "type": "perspective"
    }
  }
}

def update_plot(g, plotoptions=['Qubits', 'Holes', 'Lattice']):
    """
    Main function that updates the plot.
    """
    gnx = g.to_networkx()
    hnx = D.to_networkx()

    for i , value in enumerate(removed_nodes):
        if value == True:
            gnx.remove_node(i)

    g_nodes, g_edges = nx_to_plot(gnx, shape)
    h_nodes, h_edges = nx_to_plot(hnx, shape, index=False)
    #x_removed_nodes = [g.node_coords[j][0] for j in removed_nodes]
    #y_removed_nodes = [g.node_coords[j][1] for j in removed_nodes]
    #z_removed_nodes = [g.node_coords[j][2] for j in removed_nodes]   

    #create a trace for the edges
    trace_edges = go.Scatter3d(
        x=g_edges[0],
        y=g_edges[1],
        z=g_edges[2],
        mode='lines',
        line=dict(color='black', width=2),
        hoverinfo='none')

    #create a trace for the nodes
    trace_nodes = go.Scatter3d(
        x=g_nodes[0],
        y=g_nodes[1],
        z=g_nodes[2],
        mode='markers',
        marker=dict(symbol='circle',
                size=10,
                color='skyblue'),
        )

    trace_holes = go.Scatter3d(
        x=h_nodes[0],
        y=h_nodes[1],
        z=h_nodes[2],
        mode='markers',
        marker=dict(symbol='circle',
                size=10,
                color='green')
    )

    trace_holes_edges = go.Scatter3d(
        x=h_edges[0],
        y=h_edges[1],
        z=h_edges[2],
        mode='lines',
        line=dict(color='forestgreen', width=2),
        hoverinfo='none'
    )

    if 'Qubits' in plotoptions:
        trace_nodes.visible = True
        trace_edges.visible = True
    else:
        trace_nodes.visible = 'legendonly'
        trace_edges.visible = 'legendonly'

    if 'Holes' in plotoptions:
        trace_holes.visible = True
        trace_holes_edges.visible = True
    else:
        trace_holes.visible = 'legendonly'
        trace_holes_edges.visible = 'legendonly'



    #Include the traces we want to plot and create a figure
    data = [trace_nodes, trace_edges, trace_holes, trace_holes_edges]
    if lattice:
        if 'Lattice' in plotoptions:
            lattice.visible = True
        else:
            lattice.visible = 'legendonly'
        data.append(lattice)
    if lattice_edges:
        if 'Lattice' in plotoptions:
            lattice_edges.visible = True
        else:
            lattice_edges.visible = 'legendonly'
        data.append(lattice_edges)
        
    fig = go.Figure(data=data)
    fig.layout.height = 600
    fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0), scene_camera=camera_state["scene.camera"], legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
    ))
    return fig

f = update_plot(G)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Graph(
        id='basic-interactions',
        figure=f
    ),
    dcc.Store(id='draw-plot'),

    html.Div(className='row', children=[
        html.Div([
            dcc.Markdown(d("""
                **Hover Data**

                Mouse over values in the graph.
            """)),
            html.Pre(id='hover-data', style=styles['pre']),
            dcc.Markdown(d("""
                **Zoom and Relayout Data**

                Click and drag on the graph to zoom or click on the zoom
                buttons in the graph's menu bar.
                Clicking on legend items will also fire
                this event.
            """)),
            html.Pre(id='relayout-data', style=styles['pre']),
        ], className='three columns'),

        html.Div([
            dcc.Markdown(d("""
                **Move Log**

                Click on points in the graph.
            """)),
            html.Button('Undo', id='undo'), 
            html.Button('Run Algorithm 1', id='alg1'), 
            html.Button('Find Lattice', id='findlattice'),
            html.Button('Run Algorithm 2', id='alg2'), 
            html.Button('Repair Lattice', id='repair'),
            html.Button('Run Alg 3', id='alg3'),
            html.Pre(id='click-data', style=styles['pre'])], className='three columns'),

        html.Div([
        html.Div(id='ui'),
        dcc.Markdown(d("""
        **Select Measurement Basis**

        Click to select the type of measurement. Click points in the graph to apply measurement.
        """)),
        dcc.RadioItems(['Z', 'Y', 'X', 'Z:Hole'], 'Z', id='radio-items', inline=True),
        dcc.Markdown(d("""
        **Select display options**
        """)),
        dcc.Checklist(
            ['Qubits', 'Holes', 'Lattice'],
            ['Qubits', 'Holes', 'Lattice'],
        id='plotoptions'),
        ], className='three columns'),



        html.Div([    
            html.Div([
            dcc.Markdown(d("""
            **Reset Graph State.**

            Choose cube dimensions as well as a seed. If no seed, will use a random seed.
            """)),
            dcc.Slider(1, 15, step=1, value=xmax, tooltip={"placement": "bottom", "always_visible": True}, id="xmax"),
            dcc.Slider(1, 15, step=1, value=ymax, tooltip={"placement": "bottom", "always_visible": True}, id="ymax"),
            dcc.Slider(1, 15, step=1, value=zmax, tooltip={"placement": "bottom", "always_visible": True}, id="zmax"),
            html.Button('Reset Grid', id='reset'), ]),
            dcc.Markdown(d("""
            **Damage the Grid.**

            Select a probability p to randomly remove nodes.
            """)),
            dcc.Slider(0, 0.3, step=0.03, value=p, tooltip={"placement": "bottom", "always_visible": True}, id="prob"),
            html.Div([html.Button('Damage Grid', id='reset-seed'),
            dcc.Input(id='load-graph-seed', type="number", placeholder="Seed"),]),
            html.Div(
                [dcc.Markdown(d("""
                **Load Graph State**

                Paste data to load a graph state.
                """)),
                dcc.Input(id='load-graph-input', type="text", placeholder="Load Graph State"),
                html.Button('Load Graph', id='load-graph-button')]
           )
        ], className='three columns'),
    ])
])


@app.callback(
    Output('hover-data', 'children'),
    [Input('basic-interactions', 'hoverData')])
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)


@app.callback(
    Output('click-data', 'children'),
    Output('draw-plot','data'),
    Output('ui', 'children', allow_duplicate=True),
    Input('basic-interactions', 'clickData'), State('radio-items', 'value'), State('click-data', 'children'), prevent_initial_call = True)
def display_click_data(clickData, measurementChoice, clickLog):
    global removed_nodes, move_list
    if not clickData:
        return dash.no_update, dash.no_update
    point = clickData["points"][0]
    # Do something only for a specific trace
    if point["curveNumber"] > 0 or 'x' not in point:
        return dash.no_update, dash.no_update, ""
    else: 
        i = get_node_index(point['x'], point['y'], point['z'], shape)
        # Update the plot based on the node clicked
        if measurementChoice == 'Z:Hole':
            D.add_node(i)
            measurementChoice = 'Z' # Handle it as if it was Z measurement
        if removed_nodes[i] == False:
            removed_nodes[i] = True
            G.handle_measurements(i, measurementChoice)
            move_list.append([i, measurementChoice])
            ui = f"Clicked on {i} at {get_node_coords(i, shape)}"
        time.sleep(0.1)
        log.append(f"{i}, {measurementChoice}; ")
        log.append(html.Br())    
        return html.P(log), i, ui



@app.callback(
    Output('relayout-data', 'children'),
    [Input('basic-interactions', 'relayoutData')],
    State('relayout-data', 'children'))
def display_relayout_data(relayoutData, state):
    global camera_state
    if relayoutData and "scene.camera" in relayoutData:
        camera_state = relayoutData
        return json.dumps(relayoutData, indent=2)
    else:
        return state


@app.callback(
    Output('ui', 'children', allow_duplicate=True),
    Input('radio-items', 'value'), prevent_initial_call = True)
def update_output(value):
    return 'You have selected "{}" basis'.format(value)


@app.callback(
    Output('basic-interactions', 'figure', allow_duplicate=True),
    Output('click-data', 'children', allow_duplicate=True),
    Output('ui', 'children', allow_duplicate=True),
    Input('reset', 'n_clicks'),
    State('xmax', 'value'),
    State('ymax', 'value'),
    State('zmax', 'value'),
    prevent_initial_call=True)
def reset_grid(input, xslider, yslider, zslider, move_list_reset = True):
    global G, removed_nodes, log, move_list, lattice, lattice_edges, connected_cubes
    global shape, xmax, ymax, zmax, xoffset, yoffset, zoffset

    xmax = int(xslider)
    ymax = int(yslider)
    zmax = int(zslider)
    shape = [xmax, ymax, zmax]
    
    G = Grid(shape)
    removed_nodes = np.zeros(xmax*ymax*zmax, dtype=bool)
    fig = update_plot(G)
    log = []
    if move_list_reset:
        global D
        D = Holes(shape)
        move_list = []
        lattice = None
        lattice_edges = None
        connected_cubes = None
        xoffset = None
        yoffset = None
        zoffset = None
    # Make sure the view/angle stays the same when updating the figure        
    return fig, log, "Created grid of shape {}".format(shape)

@app.callback(
    Output('click-data', 'children', allow_duplicate=True),
    Output('draw-plot', 'data', allow_duplicate=True),
    Output('ui', 'children', allow_duplicate=True),
    Input('reset-seed', 'n_clicks'),
    State('load-graph-seed', "value"),
    State('prob', "value"),
    prevent_initial_call=True)
def reset_seed(nclicks, seed_input, prob):
    """
    Randomly measure qubits.
    """
    global D, p
    p = prob
    
    D = Holes(shape)
    if seed_input:
        # The user has inputted a seed
        random.seed(int(seed_input))
        print(f'Loaded seed : {seed_input}, p = {p}')
        ui = "Loaded seed : {}, p = {}".format(seed_input, p)
    else:
        # Use a random seed. 
        random.seed()
        print(f'Loaded seed : {seed}, p = {p}')
        ui = "Loaded seed : None, p = {}, shape = {}".format(p, shape)
    # p is the probability of losing a qubit

    measurementChoice = 'Z'
    
    for i in range(xmax*ymax*zmax):
        if random.random() < p:
            if removed_nodes[i] == False:
                removed_nodes[i] = True
                G.handle_measurements(i, measurementChoice)
                log.append(f"{i}, {measurementChoice}; ")
                log.append(html.Br())
                move_list.append([i, measurementChoice])
                D.add_node(i)
    D.add_edges()
    return log, 1, ui

@app.callback(
    Output('click-data', 'children', allow_duplicate=True),
    Output('draw-plot', 'data', allow_duplicate=True),
    Output('ui', 'children', allow_duplicate=True),
    Input('load-graph-button', 'n_clicks'),
    State('load-graph-input', "value"),
    prevent_initial_call=True)
def load_graph_from_string(n_clicks, input_string):
    reset_grid(n_clicks, xmax, ymax, zmax)

    result = process_string(input_string)

    for i, measurementChoice in result:
        removed_nodes[i] = True
        G.handle_measurements(i, measurementChoice)
        log.append(f"{i}, {measurementChoice}; ")
        log.append(html.Br())
        move_list.append([i, measurementChoice])
    return log, 1, 'Graph loaded!'

def process_string(input_string):
    input_string = input_string.replace(" ", "")
    input_string = input_string[:-1]

    # Split the string into outer lists
    outer_list = input_string.split(";")

    # Split each inner string into individual elements
    result = [inner.split(",") for inner in outer_list]
    for inner in result:
        inner[0] = int(inner[0])
    return result

@app.callback(
    Output('basic-interactions', 'figure', allow_duplicate=True),
    Input('draw-plot', 'data'),
    Input('plotoptions', 'value'),
    State('basic-interactions', 'relayoutData'),
    
    prevent_initial_call=True)
def draw_plot(data, plotoptions, relayoutData):
    """
    Called when ever the plot needs to be drawn.
    """
    fig = update_plot(G, plotoptions=plotoptions)
    # Make sure the view/angle stays the same when updating the figure
    # fig.update_layout(scene_camera=camera_state["scene.camera"])
    return fig

@app.callback(
    Output('click-data', 'children', allow_duplicate=True),
    Output('draw-plot', 'data', allow_duplicate=True),
    Output('ui', 'children', allow_duplicate=True),
    Input('undo', 'n_clicks'),
    prevent_initial_call=True)
def undo_move(n_clicks):
    if move_list:
        reset_grid(n_clicks, xmax, ymax, zmax, move_list_reset=False)
        
        undo = move_list.pop(-1)
        for move in move_list:
            i, measurementChoice = move
            removed_nodes[i] = True
            G.handle_measurements(i, measurementChoice)
            log.append(f"{i}, {measurementChoice}; ")
            log.append(html.Br())
        return log, 1, f'Undo {undo}'
    else:
        pass

@app.callback(
    Output('click-data', 'children', allow_duplicate=True),
    Output('draw-plot', 'data', allow_duplicate=True),
    Output('ui', 'children', allow_duplicate=True),
    Input('alg1', 'n_clicks'),
    prevent_initial_call=True)
def algorithm1(nclicks):
    holes = D.graph.nodes
    hole_locations = np.zeros(8)
    global xoffset, yoffset, zoffset, removed_nodes

    #counting where the holes are
    for h in holes:
        x, y, z = h
        for zoffset in range(2):
            for yoffset in range(2):
                for xoffset in range(2):
                    if ((x + xoffset) % 2 == (z + zoffset) % 2) and ((y + yoffset) % 2 == (z + zoffset) % 2):
                        hole_locations[xoffset+yoffset*2 + zoffset*4] += 1
    
    print(hole_locations)
    
    xoffset = np.argmax(hole_locations) % 2
    yoffset = np.argmax(hole_locations) // 2
    zoffset = np.argmax(hole_locations) // 4
    

    print(f"xoffset, yoffset, zoffset = {(xoffset, yoffset, zoffset)}")

    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                if ((x + xoffset) % 2 == (z + zoffset) % 2) and ((y + yoffset) % 2 == (z + zoffset) % 2):
                    i = get_node_index(x, y, z, shape)
                    if removed_nodes[i] == False:
                        G.handle_measurements(i, 'Z')
                        log.append(f"{i}, Z; ")
                        log.append(html.Br())
                        removed_nodes[i] = True
                        move_list.append([i, 'Z']) 
    
    global cubes, n_cubes
    cubes, n_cubes = D.findlattice(removed_nodes, xoffset, yoffset, zoffset)
    ui = f'Alg 1: Found {int(n_cubes[0])} unit cells. Offsets = {(xoffset, yoffset, zoffset)}'

    return log, 1, ui

@app.callback(
    Output('click-data', 'children', allow_duplicate=True),
    Output('draw-plot', 'data', allow_duplicate=True),
    Output('ui', 'children', allow_duplicate=True),
    Input('findlattice', 'n_clicks'),
    prevent_initial_call=True)
def findlattice(nclicks):
    """
    Returns:
    """
    global cubes, n_cubes, lattice, lattice_edges

    try:
        if xoffset == None:
            # cubes, n_cubes is not defined and this is because we didnt compute the offsets.
            ui = "FindLattice: Run algorithm 1 first."
            return log, 1, ui

        if n_cubes is None:
            cubes, n_cubes =  D.findlattice(removed_nodes, xoffset, yoffset, zoffset)
        #assert len(defect_box) == len(measurements_list)

        click_number = nclicks % (len(cubes))

        if len(cubes) > 0:
            C = nx.Graph()
            C.add_node(tuple(cubes[click_number][0, :]))

            X = D.connected_cube_to_nodes(C)
            
            nodes, edges = nx_to_plot(X, shape=shape, index=False)

            lattice = go.Scatter3d(
            x=nodes[0],
            y=nodes[1],
            z=nodes[2],
            mode='markers',
            line=dict(color='blue', width=2),
            hoverinfo='none'
            )

            lattice_edges = go.Scatter3d(
            x=edges[0],
            y=edges[1],
            z=edges[2],
            mode='lines',
            line=dict(color='blue', width=2),
            hoverinfo='none'
            )
            ui = f'FindLattice: Displaying {click_number+1}/{len(cubes)} unit cells found for p = {p}, shape = {shape}'
    except NameError:
        # cubes, n_cubes is not defined and this is because we didnt compute the offsets.
        ui = "FindLattice: Run algorithm 1 first."
    return log, 1, ui

@app.callback(
    Output('click-data', 'children', allow_duplicate=True),
    Output('draw-plot', 'data', allow_duplicate=True),
    Output('ui', 'children', allow_duplicate=True),
    Input('alg2', 'n_clicks'),
    prevent_initial_call=True)
def algorithm2(nclicks):
    global lattice, lattice_edges, connected_cubes
    try:
        if xoffset == None:
            # cubes, n_cubes is not defined and this is because we didnt compute the offsets.
            ui = "FindLattice: Run algorithm 1 first."
            return log, 1, ui

        C = D.build_centers_graph(cubes)
        connected_cubes = D.findconnectedlattice(C)
        for i in connected_cubes:
            print(i, len(connected_cubes))
        
        if len(connected_cubes) > 0:
            click_number = nclicks % (len(connected_cubes))
            X = D.connected_cube_to_nodes(connected_cubes[click_number])
            
            nodes, edges = nx_to_plot(X, shape=shape, index=False)

            lattice = go.Scatter3d(
            x=nodes[0],
            y=nodes[1],
            z=nodes[2],
            mode='markers',
            line=dict(color='blue', width=2),
            hoverinfo='none'
            )

            lattice_edges = go.Scatter3d(
            x=edges[0],
            y=edges[1],
            z=edges[2],
            mode='lines',
            line=dict(color='blue', width=2),
            hoverinfo='none'
            )
            ui = f"Alg 2: Displaying {click_number+1}/{len(connected_cubes)}, unit cells = {len(connected_cubes[click_number].nodes)}, edges = {len(connected_cubes[click_number].edges)}"    
        else:
            ui = f"Alg 2: No cubes found"
    except TypeError:
        ui = "Alg 2: Run Algorithm 1 first."
    except NameError:
        ui = "Alg 2: Run Algorithm 1 first."
    return log, 2, ui

@app.callback(
    Output('click-data', 'children', allow_duplicate=True),
    Output('draw-plot', 'data', allow_duplicate=True),
    Output('ui', 'children', allow_duplicate=True),
    Input('alg3', 'n_clicks'),
    prevent_initial_call=True)
def algorithm3(nclicks):
    global lattice, lattice_edges, path_clicks

    gnx = G.to_networkx()

    removed_nodes_reshape = removed_nodes.reshape((xmax, ymax, zmax))
    
    zeroplane = removed_nodes_reshape[:, :, 0]
    zmaxplane = removed_nodes_reshape[:, :, zmax-1]

    x = np.argwhere(zeroplane == 0) #This is the coordinates of all valid node in z = 0
    y = np.argwhere(zmaxplane == 0) #This is the coordinates of all valid node in z = L


    path = None
    while path is None:
        try:
            i = get_node_index(*x[path_clicks % len(x)], 0, shape)
            j = get_node_index(*y[path_clicks // len(x)], zmax-1, shape)
            path = nx.shortest_path(gnx, i, j)
        except nx.exception.NetworkXNoPath:
            ui = "No path."
            print(f'no path, {i}, {j}')
        finally:
            path_clicks += 1

    nodes, edges = path_to_plot(path, shape)
        
    lattice = go.Scatter3d(
    x=nodes[0],
    y=nodes[1],
    z=nodes[2],
    mode='markers',
    line=dict(color='blue', width=2),
    hoverinfo='none'
    )

    lattice_edges = go.Scatter3d(
    x=edges[0],
    y=edges[1],
    z=edges[2],
    mode='lines',
    line=dict(color='blue', width=2),
    hoverinfo='none'
    )

    ui = "Alg 3 ran"
    return log, 1, ui
    
    
    

@app.callback(
    Output('click-data', 'children', allow_duplicate=True),
    Output('draw-plot', 'data', allow_duplicate=True),
    Output('ui', 'children', allow_duplicate=True),
    Input('repair', 'n_clicks'),
    prevent_initial_call=True)
def repairgrid(nclicks):
    
    repairs, failures = D.repair_grid(p)
    
    reset_grid(nclicks, xmax, ymax, zmax, move_list_reset=False)
    for f in failures:
        i = get_node_index(*f, shape)
        removed_nodes[i] = True
        G.handle_measurements(i, 'Z')
        log.append(f"{i}, Z; ")
        log.append(html.Br())
        move_list.append([i, 'Z'])
    
    
    if len(repairs)+len(failures) > 0:
        rate = len(repairs)/(len(repairs)+len(failures))
        ui = f'Repairs = {len(repairs)}, Failures = {len(failures)} Repair Rate = {rate:.2f}, Holes = {np.sum(removed_nodes)}, peff={np.sum(removed_nodes)/(xmax*ymax*zmax)}'
    else:
        ui = 'All qubits repaired!'
    return log, 2, ui


app.run_server(debug=True, use_reloader=False)