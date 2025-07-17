import dash
from dash import dcc, html, Input, Output, callback, State
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import time

# Initialize the Dash app
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

# Generate synthetic data for demonstration
np.random.seed(42)
n_timesteps = 100
n_features = 50

# Simulate data for different layers
def generate_layer_data(layer_name, n_timesteps, n_features):
    """Generate synthetic neural timeseries data for a layer"""
    # Create some structure in the data
    t = np.linspace(0, 4*np.pi, n_timesteps)
    
    # Different patterns for different layers
    if layer_name == "Input":
        # More chaotic input patterns
        base_pattern = np.sin(t)[:, np.newaxis] + 0.5 * np.cos(2*t)[:, np.newaxis]
        noise_scale = 0.8
    elif layer_name == "SSM Block 1":
        # Smoother, filtered patterns
        base_pattern = np.sin(t)[:, np.newaxis] + 0.3 * np.cos(3*t)[:, np.newaxis]
        noise_scale = 0.4
    elif layer_name == "SSM Block 2":
        # More structured patterns
        base_pattern = 0.7 * np.sin(t)[:, np.newaxis] + 0.4 * np.sin(0.5*t)[:, np.newaxis]
        noise_scale = 0.3
    else:  # SSM Block 3
        # Highly structured output
        base_pattern = 0.5 * np.sin(t)[:, np.newaxis] + 0.3 * np.cos(t)[:, np.newaxis]
        noise_scale = 0.2
    
    # Generate correlated features
    data = np.random.randn(n_timesteps, n_features) * noise_scale
    
    # Add structure to first few components
    for i in range(min(5, n_features)):
        data[:, i] += base_pattern.flatten() * (1 - i * 0.1)
        if i > 0:
            data[:, i] += 0.3 * data[:, i-1]  # Add some correlation
    
    return data

# Generate data for all layers
layers_data = {
    "Input": generate_layer_data("Input", n_timesteps, n_features),
    "SSM Block 1": generate_layer_data("SSM Block 1", n_timesteps, n_features),
    "SSM Block 2": generate_layer_data("SSM Block 2", n_timesteps, n_features),
    "SSM Block 3": generate_layer_data("SSM Block 3", n_timesteps, n_features)
}

# Compute PCA for each layer
pca_results = {}
for layer_name, data in layers_data.items():
    pca = PCA(n_components=3)
    pca_coords = pca.fit_transform(data)
    pca_results[layer_name] = {
        'coordinates': pca_coords,
        'explained_variance': pca.explained_variance_ratio_,
        'pca_object': pca
    }

# Network visualization function
def create_network_diagram(selected_layer):
    """Create a clickable network diagram using HTML buttons styled as network blocks"""
    
    # Define styles for selected/unselected states
    def get_button_style(layer_name, is_selected):
        base_style = {
            'width': '140px',
            'height': '50px',
            'margin': '10px',
            'fontSize': '12px',
            'fontFamily': 'Helvetica, sans-serif',
            'fontWeight': 'bold',
            'border': '2px solid #333',
            'borderRadius': '20px',
            'cursor': 'pointer',
            'transition': 'all 0.3s ease',
            'display': 'block'
        }
        
        if is_selected:
            colors = {
                "Input": "#FF6B6B",
                "SSM Block 1": "#4ECDC4",
                "SSM Block 2": "#45B7D1",
                "SSM Block 3": "#96CEB4"
            }
            base_style.update({
                'backgroundColor': colors[layer_name],
                'color': '#333',
                'transform': 'scale(1.05)',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.2)'
            })
        else:
            colors = {
                "Input": "#FFB3B3",
                "SSM Block 1": "#A8E6E3",
                "SSM Block 2": "#A3D5E8",
                "SSM Block 3": "#CBE7D4"
            }
            base_style.update({
                'backgroundColor': colors[layer_name],
                'color': '#333'
            })
        
        return base_style
    
    return html.Div([
        html.H4("State Space Model Architecture", 
               style={'textAlign': 'center', 'marginBottom': '20px',
                      'fontFamily': 'Helvetica, sans-serif', 'color': '#333'}),
        
        html.Div([
            # Input Layer
            html.Button(
                "Input Layer",
                id='input-btn',
                n_clicks=0,
                style=get_button_style("Input", selected_layer == "Input")
            ),
            
            # Arrow down
            html.Div("↓", style={
                'textAlign': 'center', 'fontSize': '24px', 'color': '#333',
                'fontFamily': 'Helvetica, sans-serif', 'margin': '5px 0'
            }),
            
            # SSM Block 1
            html.Button(
                "SSM Block 1",
                id='ssm1-btn',
                n_clicks=0,
                style=get_button_style("SSM Block 1", selected_layer == "SSM Block 1")
            ),
            
            # Arrow down
            html.Div("↓", style={
                'textAlign': 'center', 'fontSize': '24px', 'color': '#333',
                'fontFamily': 'Helvetica, sans-serif', 'margin': '5px 0'
            }),
            
            # SSM Block 2
            html.Button(
                "SSM Block 2",
                id='ssm2-btn',
                n_clicks=0,
                style=get_button_style("SSM Block 2", selected_layer == "SSM Block 2")
            ),
            
            # Arrow down
            html.Div("↓", style={
                'textAlign': 'center', 'fontSize': '24px', 'color': '#333',
                'fontFamily': 'Helvetica, sans-serif', 'margin': '5px 0'
            }),
            
            # SSM Block 3
            html.Button(
                "SSM Block 3",
                id='ssm3-btn',
                n_clicks=0,
                style=get_button_style("SSM Block 3", selected_layer == "SSM Block 3")
            )
        ], style={
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'center',
            'padding': '20px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '15px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'
        })
    ], style={'width': '100%', 'maxWidth': '250px'})

# App layout
app.layout = html.Div([
    html.H1("PCA State Trajectories", 
            style={'textAlign': 'center', 'marginBottom': '30px', 
                   'fontFamily': 'Helvetica, sans-serif', 'color': '#2c3e50'}),
    
    html.Div([
        # Left side - Network diagram
        html.Div([
            # Initialize network diagram with buttons in the layout
            create_network_diagram('Input')
        ], style={
            'width': '25%',
            'paddingRight': '20px',
            'display': 'flex',
            'justifyContent': 'center'
        }, id='network-container'),
        
        # Right side - Plot and controls
        html.Div([
            # Layer info
            html.Div(id='layer-info', style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            # 3D plot
            html.Div([
                dcc.Graph(id='trajectory-plot', style={'height': '600px'})
            ]),
            
            # Animation controls
            html.Div([
                html.Div([
                    html.Button('Play', id='play-button', n_clicks=0, 
                              style={'marginRight': '10px', 'padding': '10px 20px',
                                    'fontSize': '16px', 'fontFamily': 'Helvetica, sans-serif',
                                    'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                                    'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Button('Reset', id='reset-button', n_clicks=0,
                              style={'padding': '10px 20px', 'fontSize': '16px', 
                                    'fontFamily': 'Helvetica, sans-serif',
                                    'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                                    'borderRadius': '5px', 'cursor': 'pointer'}),
                ], style={'textAlign': 'center', 'marginBottom': '20px'}),
                
                html.Div([
                    html.Label('Animation Speed:', style={'fontFamily': 'Helvetica, sans-serif', 'marginRight': '10px'}),
                    dcc.Slider(
                        id='speed-slider',
                        min=50,
                        max=500,
                        value=100,
                        marks={50: '0.5x', 100: '1x', 200: '2x', 500: '5x'},
                        step=50,
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '50%', 'margin': '0 auto'})
            ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px',
                     'marginTop': '20px'})
        ], style={'width': '75%'})
    ], style={'display': 'flex', 'padding': '20px', 'maxWidth': '1400px', 'margin': '0 auto'}),
    
    # Hidden divs for storing data
    html.Div(id='selected-layer', children='Input', style={'display': 'none'}),
    html.Div(id='animation-frame', children='0', style={'display': 'none'}),
    html.Div(id='is-playing', children='false', style={'display': 'none'}),
    
    # Interval for animation
    dcc.Interval(id='animation-interval', interval=100, n_intervals=0, disabled=True)
], style={'fontFamily': 'Helvetica, sans-serif'})

# Callback to update network diagram
@app.callback(
    Output('network-container', 'children'),
    [Input('selected-layer', 'children')]
)
def update_network_diagram(selected_layer):
    return create_network_diagram(selected_layer)

# Callback for layer selection
@app.callback(
    Output('selected-layer', 'children'),
    [Input('input-btn', 'n_clicks'),
     Input('ssm1-btn', 'n_clicks'),
     Input('ssm2-btn', 'n_clicks'),
     Input('ssm3-btn', 'n_clicks')],
    prevent_initial_call=True
)
def update_selected_layer(input_clicks, ssm1_clicks, ssm2_clicks, ssm3_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 'Input'
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    layer_mapping = {
        'input-btn': 'Input',
        'ssm1-btn': 'SSM Block 1',
        'ssm2-btn': 'SSM Block 2',
        'ssm3-btn': 'SSM Block 3'
    }
    
    return layer_mapping.get(button_id, 'Input')

# Callback to update layer info
@app.callback(
    Output('layer-info', 'children'),
    [Input('selected-layer', 'children')]
)
def update_layer_info(selected_layer):
    if selected_layer in pca_results:
        explained_var = pca_results[selected_layer]['explained_variance']
        return html.Div([
            html.H4(f"PCA Analysis - {selected_layer}", 
                   style={'color': '#2c3e50', 'fontFamily': 'Helvetica, sans-serif'}),
            html.P(f"Explained Variance: PC1: {explained_var[0]:.2%}, PC2: {explained_var[1]:.2%}, PC3: {explained_var[2]:.2%}",
                   style={'color': '#7f8c8d', 'fontFamily': 'Helvetica, sans-serif'})
        ])
    return ""

# Callback to update trajectory plot
@app.callback(
    Output('trajectory-plot', 'figure'),
    [Input('selected-layer', 'children'),
     Input('animation-frame', 'children')]
)
def update_trajectory_plot(selected_layer, animation_frame):
    if selected_layer not in pca_results:
        return go.Figure()
    
    coords = pca_results[selected_layer]['coordinates']
    frame_idx = int(animation_frame)
    
    # Create the trajectory trace
    fig = go.Figure()
    
    # Full trajectory (faded)
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='lines',
        line=dict(color='rgba(128, 128, 128, 0.3)', width=2),
        name='Full Trajectory',
        showlegend=False
    ))
    
    # Animated trajectory (up to current frame)
    if frame_idx > 0:
        fig.add_trace(go.Scatter3d(
            x=coords[:frame_idx, 0], 
            y=coords[:frame_idx, 1], 
            z=coords[:frame_idx, 2],
            mode='lines+markers',
            line=dict(color='#3498db', width=4),
            marker=dict(size=3, color='#3498db'),
            name='Current Trajectory',
            showlegend=False
        ))
    
    # Current position
    if frame_idx > 0:
        fig.add_trace(go.Scatter3d(
            x=[coords[frame_idx-1, 0]], 
            y=[coords[frame_idx-1, 1]], 
            z=[coords[frame_idx-1, 2]],
            mode='markers',
            marker=dict(size=8, color='#e74c3c'),
            name='Current Position',
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'3D PCA Trajectory - {selected_layer}',
            font=dict(family='Helvetica, sans-serif', size=20, color='#2c3e50')
        ),
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            xaxis=dict(title=dict(font=dict(family='Helvetica, sans-serif'))),
            yaxis=dict(title=dict(font=dict(family='Helvetica, sans-serif'))),
            zaxis=dict(title=dict(font=dict(family='Helvetica, sans-serif'))),
            bgcolor='rgba(240, 240, 240, 0.1)'
        ),
        font=dict(family='Helvetica, sans-serif'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=600
    )
    
    return fig

# Animation control callbacks
@app.callback(
    [Output('is-playing', 'children'),
     Output('play-button', 'children'),
     Output('animation-interval', 'disabled')],
    [Input('play-button', 'n_clicks'),
     Input('reset-button', 'n_clicks')],
    [State('is-playing', 'children')]
)
def control_animation(play_clicks, reset_clicks, is_playing):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 'false', 'Play', True
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'play-button':
        if is_playing == 'false':
            return 'true', 'Pause', False
        else:
            return 'false', 'Play', True
    elif button_id == 'reset-button':
        return 'false', 'Play', True
    
    return is_playing, 'Play', True

# Animation frame update
@app.callback(
    [Output('animation-frame', 'children'),
     Output('animation-interval', 'interval')],
    [Input('animation-interval', 'n_intervals'),
     Input('reset-button', 'n_clicks'),
     Input('speed-slider', 'value')],
    [State('animation-frame', 'children'),
     State('is-playing', 'children')]
)
def update_animation_frame(n_intervals, reset_clicks, speed, current_frame, is_playing):
    ctx = dash.callback_context
    
    if ctx.triggered and ctx.triggered[0]['prop_id'].split('.')[0] == 'reset-button':
        return '0', 1000 // speed
    
    if is_playing == 'true':
        new_frame = (int(current_frame) + 1) % n_timesteps
        return str(new_frame), 1000 // speed
    
    return current_frame, 1000 // speed

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)