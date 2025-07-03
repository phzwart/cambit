import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import numpy as np
import json
import colorsys

def hex_to_rgba(hex_color, alpha=1.0):
    """
    Converts a hex color string to an RGBA color string.

    Parameters:
    hex_color (str): The color to convert, in hex format.
    alpha (float, optional): The alpha (opacity) value of the color. Default is 1.0.

    Returns:
    str: The RGBA color string.
    """
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


def generate_colors(num_colors):
    """
    Generates a list of color codes.

    Parameters:
    num_colors (int): The number of color codes to generate.

    Returns:
    list: A list of color codes.
    """
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        lightness = 0.5
        saturation = 0.8
        r, g, b = [int(x * 255) for x in colorsys.hls_to_rgb(hue, lightness, saturation)]
        colors.append(f'#{r:02x}{g:02x}{b:02x}')
    return colors


def generate_scattergl_plot(x_coords, y_coords, labels, label_to_string_map, show_legend=False, custom_indices=None):
    """
    Generates a Scattergl plot.

    Parameters:
    x_coords (list): The x-coordinates of the points.
    y_coords (list): The y-coordinates of the points.
    labels (list): The labels of the points.
    label_to_string_map (dict): A mapping from labels to strings.
    show_legend (bool, optional): Whether to show a legend. Default is False.
    custom_indices (list, optional): Custom indices for the points. Default is None.

    Returns:
    go.Figure: The generated Scattergl plot.
    """
    # Create a set of unique labels
    unique_labels = set(labels)

    # Create a trace for each unique label
    traces = []
    for label in unique_labels:
        # Find the indices of the points with the current label
        trace_indices = [i for i, l in enumerate(labels) if l == label]
        trace_x = [x_coords[i] for i in trace_indices]
        trace_y = [y_coords[i] for i in trace_indices]

        # Always use custom_indices if provided, otherwise use trace indices
        # custom_indices should contain the global dataset indices
        if custom_indices is not None:
            trace_custom_indices = [custom_indices[i] for i in trace_indices]
        else:
            trace_custom_indices = trace_indices

        trace = go.Scattergl(
            x=trace_x,
            y=trace_y,
            customdata=np.array(trace_custom_indices).reshape(-1, 1),
            mode='markers',
            name=str(label_to_string_map[label])
        )
        print(f"Trace {label}: customdata shape = {trace.customdata.shape}, values = {trace.customdata.flatten()[:5]}...")  # Debug
        traces.append(trace)

    # Create the plot with the scatter plot traces
    fig = go.Figure(data=traces)
    
    # Add layout constraints to prevent canvas expansion
    fig.update_layout(
        autosize=True,
        height=600,  # Fixed height
        width=None,  # Auto width
        margin=dict(l=50, r=50, t=50, b=50),  # Fixed margins
        xaxis=dict(
            constrain='domain',  # Constrain to domain
            autorange=True
        ),
        yaxis=dict(
            scaleanchor="x",  # Scale with x-axis
            scaleratio=1,     # Maintain aspect ratio
            constrain='domain',  # Constrain to domain
            autorange=True
        )
    )
    
    if show_legend:
        fig.update_layout(
            legend=dict(
                x=0,
                y=1,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(255, 255, 255, 0.9)',
                orientation='h'
            )
        )
    return fig

def generate_scatter_data(latent_vectors,
                          cluster_selection=-1,
                          clusters=None,
                          cluster_names=None,
                          label_selection=-2,
                          labels=None,
                          label_names=None,
                          color_by=None,
                          ):
    """
    Generate data for a plot according to the provided selection options:
    1. all clusters & all labels
    2. all clusters and selected labels
    3. all labels and selected clusters
    4. selected clusters and selected labels

    Parameters:
    latent_vectors (numpy.ndarray, Nx2, floats): [Description]
    cluster_selection (int): The cluster w want to select. Defaults to -1: all clusters
    clusters (numpy.ndarray, N, ints optional): The cluster number for each data point
    cluster_names (dict, optional): [Description]. A dictionary with cluster names
    label_selection (str, optional): Which label to select. Defaults to -2: all labels. -1 mean Unlabeled
    labels (numpy.ndarray, N, int, optional): The current labels Defaults to None.
    label_names (dict, optional): A dictionary that relates label number to name.
    color_by (str, optional): Determines if we color by label or cluster. Defaults to None.

    Returns:
    plotly.scattergl: A plot as specified.
    """
    # case:
    #  all data: cluster_selection =-1, label_selection=-2
    #  all clusters, selected labels
    #  all labels, selected clusters
    marker_dict = None
    vals_names = {}  # None
    if color_by == 'cluster':
        vals = clusters
        vals_names = cluster_names
    if color_by == 'label':
        vals_names = {value: key for key, value in label_names.items()}
        vals_names[-1] = "N/A"
        vals = labels

    if (cluster_selection == -1) & (label_selection == -2):
        # Create indices for all data points
        all_indices = np.arange(len(latent_vectors))
        scatter_data = generate_scattergl_plot(latent_vectors[:, 0],
                                               latent_vectors[:, 1],
                                               vals,
                                               vals_names,
                                               custom_indices=all_indices)
        return scatter_data

    selected_indices = None
    if (cluster_selection == -1) & (label_selection != -2):  # all clusters
        if label_selection != -1:
            label_selection = label_names[label_selection]
        selected_indices = np.where(labels == label_selection)[0]

    if (label_selection == -2) & (cluster_selection > -1):  # all clusters
        selected_indices = np.where(clusters == cluster_selection)[0]

    if (label_selection != -2) & (cluster_selection > -1):
        if label_selection != -1:
            selected_labels = label_names[label_selection]
            selected_indices = np.where((clusters == cluster_selection) & (labels == selected_labels))[0]
        else:
            selected_indices = np.where((clusters == cluster_selection))[0]

    scatter_data = generate_scattergl_plot(latent_vectors[selected_indices, 0],
                                           latent_vectors[selected_indices, 1],
                                           vals[selected_indices],
                                           vals_names,
                                           custom_indices=selected_indices)
    return scatter_data


def generate_cluster_dropdown_options(clusters):
    """
    Generates options for a cluster dropdown menu.

    Parameters:
    clusters (numpy.ndarray): The array of cluster labels.

    Returns:
    list: A list of dictionaries, each representing an option for the dropdown. Each dictionary has a 'label' key
    for the display text, and a 'value' key for the corresponding value.
    """
    unique_clusters = np.unique(clusters)
    options = [{'label': f'Cluster {cluster}', 'value': cluster} for cluster in unique_clusters if cluster != -1]
    options.insert(0, {'label': 'All', 'value': -1})
    return options

def generate_label_dropdown_options(label_names, add_all=True):
    """
    Generates options for a label dropdown menu.

    Parameters:
    label_names (dict): The mapping from labels to names.
    add_all (bool, optional): Whether to add an 'All' option. Default is True.

    Returns:
    list: A list of dictionaries, each representing an option for the dropdown. Each dictionary has a 'label' key
    for the display text, and a 'value' key for the corresponding value.
    """
    options = [{'label': f'{label_names[label]}', 'value': label} for label in label_names]
    options.insert(0, {'label': 'Unlabeled', 'value': -1})
    if add_all:
        options.insert(0, {'label': 'All', 'value': -2})
    return options


def compute_mean_std_images(selected_indices, images):
    """
    Computes the mean and standard deviation of a selection of images.

    Parameters:
    selected_indices (list): The indices of the selected images.
    images (numpy.ndarray): The array of all images.

    Returns:
    tuple: A tuple containing the mean image and the standard deviation image.
    """
    selected_images = images[selected_indices]
    mean_image = np.mean(selected_images, axis=0)
    std_image = np.std(selected_images, axis=0)
    return mean_image, std_image


# %%

def build_explorer(images,
                   latent_vectors,
                   clusters,
                   label_names,
                   assigned_labels):
    
    # Global variable to store the current trace data for index lookup
    # Make it accessible to callbacks by using a list (mutable)
    trace_data_container = [{}]
    """
    Constructs an explorer for visualizing and interacting with image data, latent vectors, and associated labels.

    Parameters:
    images (numpy.ndarray, [N, Y, X] ): An array of images. Each image is represented as a multidimensional array.
    latent_vectors (numpy.ndarray, [N,2] ): An array of latent vectors corresponding to the images.
    clusters (numpy.ndarray [N]): An array of cluster assignments for each image.
    label_names (dict): A dictionary mapping label identifiers to their respective names.
    assigned_labels (numpy.ndarray [N] ): An array of labels assigned to each image.

    """

    # Initialize the app
    app = Dash(__name__)

    cluster_names = {a: a for a in np.unique(clusters).astype(int)}

    app.layout = html.Div([
        html.Div([
            # individual image
            html.Div([
                dcc.Graph(id='heatmap-a', figure=go.Figure(go.Image()), style={'padding-bottom': '5%', 'height': '600px'}),
            ], className='column', style={'flex': '50%', 'padding': '10px'}),

            # latent plot
            html.Div([
                dcc.Graph(id='scatter-b',
                          figure=go.Figure(go.Scattergl(mode='markers')),
                          style={'padding-bottom': '5%', 'height': '600px'}),
            ], className='column', style={'flex': '50%', 'padding': '10px'}),

        ], className='row', style={'display': 'flex'}),
        html.Div([
            # control panel
            html.Div([
                # Add controls and human interactions here
                # Example: dcc.Slider(), dcc.Dropdown(), etc.

                html.Label('Select cluster:'),
                dcc.Dropdown(id='cluster-dropdown',
                             options=generate_cluster_dropdown_options(clusters),
                             value=-1),

                html.Label('Select label:'),
                dcc.Dropdown(id='label-dropdown',
                             options=generate_label_dropdown_options(label_names),
                             value=-2),

                # Add a radio button for toggling mean and standard deviation
                html.Label('Display Image Options:'),
                dcc.RadioItems(id='mean-std-toggle', options=[{'label': 'Mean', 'value': 'mean'},
                                                              {'label': 'Standard Deviation', 'value': 'sigma'}],
                               value='mean'),

                # Add a radio button for toggling coloring options
                html.Label('Scatter Colors:'),
                dcc.RadioItems(id='scatter-color', options=[{'label': 'cluster', 'value': 'cluster'},
                                                            {'label': 'label', 'value': 'label'}],
                               value='cluster'),

            ], className='column', style={'flex': '50%', 'padding-bottom': '5%'}),

            # Labeler
            # Add a new div for displaying statistics
            html.Div([

                html.Div(id='stats-div', children=[
                    html.P("Number of images selected: 0"),
                    html.P("Clusters represented: N/A"),
                    html.P("Labels represented: N/A"),
                ]),

                html.Label('Assign Label:'),
                dcc.Dropdown(id='labeler',
                             options=generate_label_dropdown_options(label_names, False),
                             value=-1),

                html.Button('Assign Labels', id='assign-labels-button'),

                html.Div(id='label-assign-output'),

            ], className='column', style={'flex': '50%', 'padding': '10px'}),

        ], className='row', style={'display': 'flex'}),

        # hidden components
        html.Div(id="scatter-update-trigger", style={"display": "none"}),
        dcc.Store(id='scatter-axis-range', storage_type='session'),
        dcc.Store(id='selected-points', storage_type='memory'),
        dcc.Store(id='selected-data-store', data=None),
        dcc.Store(id='trace-data-store', data=None),
        html.Script("""
                        document.addEventListener('DOMContentLoaded', function() {
                            document.getElementById('assign-labels-button').onclick = function() {
                                setTimeout(function() {
                                    document.getElementById('scatter-b').focus();
                                }, 100);
                            };
                        });
                    """)

    ], style={'display': 'grid', 'gridTemplateRows': '1fr 1fr', 'height': '100vh'})

    # -------------------------------------------------
    # SCATTER PLOT CALLBACKs

    @app.callback(
        [Output('scatter-b', 'figure'),
         Output('selected-data-store', 'data'),
         Output('trace-data-store', 'data')],
        Input('cluster-dropdown', 'value'),
        Input('label-dropdown', 'value'),
        Input('scatter-color', 'value'),
        State('labeler', 'value'),
        State('scatter-b', 'figure'),
        State('scatter-b', 'selectedData')
    )
    def update_scatter_plot(cluster_selection, label_selection, scatter_color, labeler_value, current_figure,
                            selected_data):
        if selected_data is not None and len(selected_data.get('points', [])) > 0:
            try:
                selected_indices = [point.get('customdata', [None])[0] for point in selected_data['points']]
                # Filter out None values
                selected_indices = [idx for idx in selected_indices if idx is not None]
            except (KeyError, IndexError, TypeError):
                selected_indices = None
        else:
            selected_indices = None

        fig = generate_scatter_data(latent_vectors,
                                     cluster_selection,
                                     clusters,
                                     cluster_names,
                                     label_selection,
                                     assigned_labels,
                                     label_names,
                                     scatter_color)
        
        # Store trace data for index lookup
        trace_data_container[0].clear()
        for i, trace in enumerate(fig.data):
            if hasattr(trace, 'customdata') and trace.customdata is not None:
                trace_data_container[0][i] = trace.customdata.flatten().tolist()
        print(f"Updated trace_data_container: {trace_data_container[0]}")  # Debug
        
        # Update the figure layout
        fig.update_layout(
            legend=dict(tracegroupgap=20),
            autosize=True,
            height=600,  # Fixed height
            width=None,  # Auto width
            margin=dict(l=50, r=50, t=50, b=50),  # Fixed margins
            xaxis=dict(
                constrain='domain',  # Constrain to domain
                autorange=True
            ),
            yaxis=dict(
                scaleanchor="x",  # Scale with x-axis
                scaleratio=1,     # Maintain aspect ratio
                constrain='domain',  # Constrain to domain
                autorange=True
            )
        )
        # print(labeler_value)

        if current_figure and 'xaxis' in current_figure['layout'] and 'yaxis' in current_figure[
            'layout'] and 'autorange' in current_figure['layout']['xaxis'] and current_figure['layout']['xaxis'][
            'autorange'] is False:
            # Update the axis range with current figure's values if available and if autorange is False
            fig.update_xaxes(range=current_figure['layout']['xaxis']['range'])
            fig.update_yaxes(range=current_figure['layout']['yaxis']['range'])
        else:
            # If it's the initial figure or autorange is True, set autorange to True to fit all points in view
            fig.update_xaxes(autorange=True)
            fig.update_yaxes(autorange=True)

        if selected_indices is not None and len(selected_indices) > 0:
            # Use the selected indices to highlight the selected points in the updated figure
            for trace in fig.data:
                if trace.marker.color is not None:
                    trace.marker.color = [hex_to_rgba('grey', 0.3) if i not in selected_indices else 'red' for i in
                                          range(len(trace.marker.color))]
        return fig, selected_data, trace_data_container[0]

    # -------------------------------------------------
    # IMAGE PANEL
    @app.callback(
        Output('heatmap-a', 'figure'),
        Input('scatter-b', 'clickData'),
        Input('scatter-b', 'selectedData'),
        Input('mean-std-toggle', 'value'),
        State('heatmap-a', 'figure')
    )
    def update_panel_a(click_data, selected_data, display_option, current_figure):
        import plotly.graph_objects as go
        import numpy as np
        if selected_data is not None and len(selected_data['points']) > 0:
            try:
                # Use the stored trace data to get the correct indices
                selected_indices = []
                for point in selected_data['points']:
                    curve_number = point.get('curveNumber', 0)
                    point_index = point.get('pointIndex', 0)
                    
                    if curve_number in trace_data_container[0] and point_index < len(trace_data_container[0][curve_number]):
                        selected_index = trace_data_container[0][curve_number][point_index]
                        selected_indices.append(selected_index)
                
                if not selected_indices:
                    # Fallback to default if no valid indices
                    heatmap_data = go.Heatmap()
                else:
                    selected_images = images[selected_indices]
            except (KeyError, IndexError, TypeError) as e:
                print(f"Error processing selected data: {e}")  # Debug
                # Fallback to default heatmap with correct dimensions
                if len(images.shape) == 3:
                    # Grayscale images
                    default_data = np.random.rand(images.shape[1], images.shape[2])
                    heatmap_data = go.Heatmap(z=default_data)
                elif len(images.shape) == 4:
                    # RGB images
                    default_data = np.random.rand(images.shape[1], images.shape[2], 3)
                    default_data = np.clip(default_data * 255, 0, 255).astype(np.uint8)
                    heatmap_data = go.Image(z=default_data)
                else:
                    # Fallback
                    default_data = np.random.rand(28, 28)
                    heatmap_data = go.Heatmap(z=default_data)
            else:
                # Process the selected images
                if selected_images.ndim == 4 and selected_images.shape[-1] == 3:
                    # RGB images
                    if display_option == 'mean':
                        mean_img = np.mean(selected_images, axis=0)
                        mean_img = np.clip(mean_img * 255, 0, 255).astype(np.uint8)
                        heatmap_data = go.Image(z=mean_img)
                    elif display_option == 'sigma':
                        std_img = np.std(selected_images, axis=0)
                        std_img = np.clip((std_img / np.max(std_img)) * 255, 0, 255).astype(np.uint8)
                        heatmap_data = go.Image(z=std_img)
                    else:
                        mean_img = np.mean(selected_images, axis=0)
                        mean_img = np.clip(mean_img * 255, 0, 255).astype(np.uint8)
                        heatmap_data = go.Image(z=mean_img)
                elif selected_images.ndim == 3:
                    # Grayscale images
                    if display_option == 'mean':
                        mean_img = np.mean(selected_images, axis=0)
                        heatmap_data = go.Heatmap(z=mean_img)
                    elif display_option == 'sigma':
                        std_img = np.std(selected_images, axis=0)
                        heatmap_data = go.Heatmap(z=std_img)
                    else:
                        mean_img = np.mean(selected_images, axis=0)
                        heatmap_data = go.Heatmap(z=mean_img)
                else:
                    # Fallback
                    heatmap_data = go.Heatmap()
        elif click_data is not None and len(click_data['points']) > 0:
            try:
                # Use the stored trace data to get the correct index
                point = click_data['points'][0]
                curve_number = point.get('curveNumber', 0)
                point_index = point.get('pointIndex', 0)
                
                if curve_number in trace_data_container[0] and point_index < len(trace_data_container[0][curve_number]):
                    selected_index = trace_data_container[0][curve_number][point_index]
                    image = images[selected_index]
                    if image.ndim == 3 and image.shape[-1] == 3:
                        # RGB
                        image = np.clip(image * 255, 0, 255).astype(np.uint8)
                        heatmap_data = go.Image(z=image)
                    else:
                        heatmap_data = go.Heatmap(z=image)
                else:
                    # Fallback to default
                    heatmap_data = go.Heatmap()
            except (KeyError, IndexError, TypeError) as e:
                print(f"Error processing click data: {e}")  # Debug
                # Fallback to default heatmap with correct dimensions
                if len(images.shape) == 3:
                    # Grayscale images
                    default_data = np.random.rand(images.shape[1], images.shape[2])
                    heatmap_data = go.Heatmap(z=default_data)
                elif len(images.shape) == 4:
                    # RGB images
                    default_data = np.random.rand(images.shape[1], images.shape[2], 3)
                    default_data = np.clip(default_data * 255, 0, 255).astype(np.uint8)
                    heatmap_data = go.Image(z=default_data)
                else:
                    # Fallback
                    default_data = np.random.rand(28, 28)
                    heatmap_data = go.Heatmap(z=default_data)
        else:
            heatmap_data = go.Heatmap()

        # Determine the aspect ratio based on the shape of the heatmap_data's z-values
        aspect_x = 1
        aspect_y = 1
        if hasattr(heatmap_data, 'z') and heatmap_data['z'] is not None:
            if hasattr(heatmap_data['z'], 'size') and heatmap_data['z'].size > 0:
                if len(heatmap_data['z'].shape) == 2:
                    aspect_y, aspect_x = np.shape(heatmap_data['z'])
                elif len(heatmap_data['z'].shape) == 3:
                    aspect_y, aspect_x = np.shape(heatmap_data['z'])[:2]

        return go.Figure(
            data=heatmap_data,
            layout=dict(
                autosize=True,
                yaxis=dict(scaleanchor="x", scaleratio=aspect_y / aspect_x),
            )
        )

    # -------------------------------------------------
    # DISPLAY SELECTION STATISTICS
    @app.callback(
        Output('stats-div', 'children'),
        Input('scatter-b', 'selectedData'),
        Input('assign-labels-button', 'n_clicks'),
    )
    def update_statistics(selected_data, n_clicks):
        try:
            if selected_data is not None and len(selected_data.get('points', [])) > 0:
                try:
                    # Use the stored trace data to get the correct indices
                    selected_indices = []
                    for point in selected_data['points']:
                        curve_number = point.get('curveNumber', 0)
                        point_index = point.get('pointIndex', 0)
                        
                        if curve_number in trace_data_container[0] and point_index < len(trace_data_container[0][curve_number]):
                            selected_index = trace_data_container[0][curve_number][point_index]
                            selected_indices.append(selected_index)
                    
                    if len(selected_indices) > 0:
                        selected_clusters = clusters[selected_indices]
                        selected_labels = assigned_labels[selected_indices]

                        num_images = len(selected_indices)
                        unique_clusters = np.unique(selected_clusters)
                        unique_labels = np.unique(selected_labels)

                        # Format the clusters and labels as comma-separated strings
                        clusters_str = ", ".join(str(cluster) for cluster in unique_clusters)
                        labels_str = ", ".join(str(label_names[label]) for label in unique_labels if label in label_names)
                    else:
                        num_images = 0
                        clusters_str = "N/A"
                        labels_str = "N/A"
                except (KeyError, IndexError, TypeError):
                    num_images = 0
                    clusters_str = "N/A"
                    labels_str = "N/A"
            else:
                num_images = 0
                clusters_str = "N/A"
                labels_str = "N/A"

            return [
                html.P(f"Number of images selected: {num_images}"),
                html.P(f"Clusters represented: {clusters_str}"),
                html.P(f"Labels represented: {labels_str}"),
            ]
        except Exception as e:
            print(f"Stats error: {e}")
            return [
                html.P("Number of images selected: 0"),
                html.P("Clusters represented: N/A"),
                html.P("Labels represented: N/A"),
            ]

    @app.callback(
        Output("scatter-update-trigger", "children"),
        Input("assign-labels-button", "n_clicks"),
        State('labeler', 'value'),
        State('scatter-b', 'selectedData')
    )
    def trigger_scatter_update(n_clicks, labeler_value, selected_data):
        if n_clicks is not None:
            if n_clicks > 0:
                if selected_data is not None and len(selected_data.get('points', [])) > 0:
                    try:
                        # Use the stored trace data to get the correct indices
                        selected_indices = []
                        for point in selected_data['points']:
                            curve_number = point.get('curveNumber', 0)
                            point_index = point.get('pointIndex', 0)
                            
                            if curve_number in trace_data_container[0] and point_index < len(trace_data_container[0][curve_number]):
                                selected_index = trace_data_container[0][curve_number][point_index]
                                selected_indices.append(selected_index)
                        
                        for idx in selected_indices:
                            if labeler_value != -1:
                                assigned_labels[idx] = labeler_value
                            else:
                                assigned_labels[idx] = -1
                    except (KeyError, IndexError, TypeError):
                        pass  # Silently ignore if customdata is missing

                return n_clicks
            else:
                return n_clicks

        else:
            return n_clicks

        return n_clicks

    @app.callback(
        Output('scatter-axis-range', 'data'),
        Input('scatter-b', 'relayoutData')
    )
    def store_scatter_axis_range(relayout_data):
        if relayout_data and ('xaxis.range[0]' in relayout_data or 'yaxis.range[0]' in relayout_data):
            return {
                'x_range': [relayout_data.get('xaxis.range[0]', None), relayout_data.get('xaxis.range[1]', None)],
                'y_range': [relayout_data.get('yaxis.range[0]', None), relayout_data.get('yaxis.range[1]', None)]
            }
        return {}

    # Remove this callback as it references a non-existent ID
    # @app.callback(
    #     Output('label-assign-output', 'children'),
    #     Input('label-assign-message', 'children')
    # )
    # def update_label_assign_output(message):
    #     return message

    # Add a method to run the app with the correct syntax
    def run_app(self, mode='inline', port=8050, debug=False):
        """Run the app with the correct Dash syntax"""
        return self.run(host='127.0.0.1', port=port, debug=debug)
    
    # Add the run_app method to the app instance
    app.run_app = run_app.__get__(app)
    
    return app
