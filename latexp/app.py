import plotly.graph_objects as go
import plotly.express as px
from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import numpy as np
import json
import numpy as np
import plotly.express as px
import colorsys


def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        lightness = 0.5
        saturation = 0.8
        r, g, b = [int(x * 255) for x in colorsys.hls_to_rgb(hue, lightness, saturation)]
        colors.append(f'#{r:02x}{g:02x}{b:02x}')
    return colors


def generate_scattergl_plot(x_coords,
                            y_coords,
                            labels,
                            label_to_string_map,
                            show_legend=False,
                            custom_indices=None):
    # Create a set of unique labels
    unique_labels = set(labels)

    # Create a trace for each unique label
    traces = []
    for label in unique_labels:
        # Find the indices of the points with the current label
        trace_indices = [i for i, l in enumerate(labels) if l == label]
        trace_x = [x_coords[i] for i in trace_indices]
        trace_y = [y_coords[i] for i in trace_indices]

        if custom_indices is not None:
            trace_custom_indices = [custom_indices[i] for i in trace_indices]
        else:
            trace_custom_indices = trace_indices

        traces.append(
            go.Scattergl(
                x=trace_x,
                y=trace_y,
                customdata=np.array(trace_custom_indices).reshape(-1, 1),
                mode='markers',
                name=str(label_to_string_map[label])
            )
        )

    # Create the plot with the scatter plot traces
    fig = go.Figure(data=traces)
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
        scatter_data = generate_scattergl_plot(latent_vectors[:, 0],
                                               latent_vectors[:, 1],
                                               vals,
                                               vals_names)
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
    unique_clusters = np.unique(clusters)
    options = [{'label': f'Cluster {cluster}', 'value': cluster} for cluster in unique_clusters if cluster != -1]
    options.insert(0, {'label': 'All', 'value': -1})
    return options


def generate_label_dropdown_options(label_names, add_all=True):
    options = [{'label': f'Label {label}', 'value': label} for label in label_names]
    options.insert(0, {'label': 'Unlabeled', 'value': -1})
    if add_all:
        options.insert(0, {'label': 'All', 'value': -2})
    return options


def compute_mean_std_images(selected_indices, images):
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
    # Initialize the app
    app = JupyterDash(__name__)

    cluster_names = {a: a for a in np.unique(clusters).astype(int)}

    app.layout = html.Div([
        html.Div([
            # individual image
            html.Div([
                dcc.Graph(id='heatmap-a', figure=go.Figure(go.Heatmap()), style={'padding-bottom': '5%'}),
            ], className='column', style={'flex': '50%', 'padding': '10px'}),

            # latent plot
            html.Div([
                dcc.Graph(id='scatter-b',
                          figure=go.Figure(go.Scattergl(mode='markers')),
                          style={'padding-bottom': '5%'}),
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
        Output('scatter-b', 'figure'),
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
            selected_indices = [point['customdata'][0] for point in selected_data['points']]
        else:
            selected_indices = None

        scatter_data = generate_scatter_data(latent_vectors,
                                             cluster_selection,
                                             clusters,
                                             cluster_names,
                                             label_selection,
                                             assigned_labels,
                                             label_names,
                                             scatter_color)

        fig = go.Figure(scatter_data)
        fig.update_layout(legend=dict(tracegroupgap=20))
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

        if selected_indices is not None:
            # Use the selected indices to highlight the selected points in the updated figure
            for trace in fig.data:
                if trace.marker.color is not None:
                    trace.marker.color = [hex_to_rgba('grey', 0.3) if i not in selected_indices else 'red' for i in
                                          range(len(trace.marker.color))]
        return fig

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
        if selected_data is not None and len(selected_data['points']) > 0:
            selected_indices = [point['customdata'][0] for point in
                                selected_data['points']]  # Access customdata for the original indices
            selected_images = images[selected_indices]
            if display_option == 'mean':
                heatmap_data = go.Heatmap(z=np.mean(selected_images, axis=0))
            elif display_option == 'sigma':
                heatmap_data = go.Heatmap(z=np.std(selected_images, axis=0))
        elif click_data is not None and len(click_data['points']) > 0:
            selected_index = click_data['points'][0]['customdata'][0]  # click_data['points'][0]['pointIndex']
            heatmap_data = go.Heatmap(z=images[selected_index])
        else:
            heatmap_data = go.Heatmap()

        # Determine the aspect ratio based on the shape of the heatmap_data's z-values
        aspect_x = 1
        aspect_y = 1
        if heatmap_data['z'] is not None:
            if heatmap_data['z'].size > 0:
                aspect_y, aspect_x = np.shape(heatmap_data['z'])

        return go.Figure(
            data=heatmap_data,
            layout=dict(
                autosize=True,
                yaxis=dict(scaleanchor="x", scaleratio=aspect_y / aspect_x),
            )
        )

        return go.Figure(heatmap_data)

    # -------------------------------------------------
    # DISPLAY SELECTION STATISTICS
    @app.callback(
        Output('stats-div', 'children'),
        Input('scatter-b', 'selectedData'),
        Input('assign-labels-button', 'n_clicks'),
    )
    def update_statistics(selected_data, n_clicks):
        if selected_data is not None and len(selected_data['points']) > 0:
            selected_indices = [point['customdata'][0] for point in
                                selected_data['points']]  # Access customdata for the original indices
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

        return [
            html.P(f"Number of images selected: {num_images}"),
            html.P(f"Clusters represented: {clusters_str}"),
            html.P(f"Labels represented: {labels_str}"),
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
                if selected_data is not None and len(selected_data['points']) > 0:
                    selected_indices = [point['customdata'][0] for point in selected_data['points']]
                    for idx in selected_indices:
                        if labeler_value != -1:
                            assigned_labels[idx] = label_names[labeler_value]
                        else:
                            assigned_labels[idx] = -1

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

    @app.callback(
        Output('label-assign-output', 'children'),
        Input('label-assign-message', 'children')
    )
    def update_label_assign_output(message):
        return message

    return app
