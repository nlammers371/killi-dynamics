import plotly.express as px
import plotly.graph_objects as go
import math


def format_2d_plotly(fig, axis_labels=None, font_size=14, marker_size=6, no_markers=False,
                     theme="dark", dims=None, title="", show_gridlines=True, marker_lines=True):
    """
    Format a 2D Plotly figure (scatter plot) with consistent styling.

    Parameters:
      fig (go.Figure): A Plotly figure object.
      axis_labels (list): A list with two strings for the x- and y-axis titles.
      font_size (int): Global font size.
      marker_size (int): Marker size for traces.
      theme (str): "dark" or "light".
      dims (list): [height, width] for the figure (default: [600, 1000]).
      title (str): Plot title.

    Returns:
      A formatted Plotly figure.
    """
    # Default dimensions
    if dims is None:
        dims = [600, 1000]
    # Theme colors
    if theme == "dark":
        line_color = "white"
        text_color = "white"
        bk_color = "black"
    else:
        line_color = "black"
        text_color = "black"
        bk_color = "white"

    # Axis labels default
    if axis_labels is None:
        axis_labels = ["", ""]

    # Marker adjustments: only for traces that support markers
    for trace in fig.data:
        if hasattr(trace, 'marker') & ~no_markers:
            if marker_lines:
                trace.marker.size = marker_size
                trace.marker.line = dict(color=line_color, width=1)
            else:
                trace.marker.size = marker_size

    # Compute tick font size
    tick_font_size = font_size  # or adjust as needed

    # Axis formatting dict
    axis_format_dict = dict(
        showgrid=show_gridlines,
        zeroline=True,
        gridcolor=line_color,
        linecolor=line_color,
        zerolinecolor=line_color,
        tickfont=dict(size=tick_font_size)
    )

    # X and Y axis dicts with titles
    xaxis_format = axis_format_dict.copy()
    xaxis_format['title'] = axis_labels[0]
    yaxis_format = axis_format_dict.copy()
    yaxis_format['title'] = axis_labels[1]

    # Check for manual axis ranges
    x_range = fig.layout.xaxis.range if hasattr(fig.layout, 'xaxis') else None
    y_range = fig.layout.yaxis.range if hasattr(fig.layout, 'yaxis') else None

    if x_range is not None and y_range is not None:
        x_extent = x_range[1] - x_range[0]
        y_extent = y_range[1] - y_range[0]
        fig.update_layout(
            xaxis=dict(**xaxis_format, range=x_range, scaleanchor='y', scaleratio=x_extent / y_extent),
            yaxis=dict(**yaxis_format, range=y_range)
        )
    else:
        fig.update_layout(
            xaxis=xaxis_format,
            yaxis=yaxis_format
        )

    # General layout settings
    fig.update_layout(
        width=dims[1],
        height=dims[0],
        title=title,
        font=dict(color=text_color, family="Arial, sans-serif", size=font_size),
        plot_bgcolor=bk_color,
        paper_bgcolor=bk_color
    )

    # Remove error bars if present
    try:
        fig.update_traces(
            error_x=dict(color=line_color, width=0),
            error_y=dict(color=line_color, width=0)
        )
    except Exception:
        pass

    # Adjust colorbar position if legends present
    if any(hasattr(trace, 'marker') and getattr(trace, 'showlegend', True)
           for trace in fig.data):
        fig.update_layout(
            coloraxis_colorbar=dict(x=1, y=0, len=0.5, yanchor='bottom')
        )
    else:
        fig.update_layout(
            coloraxis_colorbar=dict(x=1, y=0.5, len=0.5)
        )

    return fig

def format_3d_plotly(fig, axis_labels=None, font_size=14, marker_size=6, show_gridlines=True, hide_axes=False,
                     aspectmode="data", eye=None, theme="dark", dims=None, title="", marker_lines=True):
    """
    Apply consistent 3D formatting to a Plotly figure. Only adjusts markers on traces that support them.
    """
    import math
    import plotly.graph_objects as go

    # Default dimensions
    if dims is None:
        dims = [600, 800]
    # Theme colors
    if theme == "dark":
        line_color = "white"
        text_color = "white"
        bk_color = "black"
    else:
        line_color = "black"
        text_color = "black"
        bk_color = "white"

    # Axis labels default
    if axis_labels is None:
        axis_labels = ["", "", ""]
    # Camera default
    if eye is None:
        eye = dict(x=1.5, y=1.5, z=1.5)

    # Marker adjustments: only for traces with a marker attribute
    for trace in fig.data:
        if hasattr(trace, 'marker'):
            if marker_lines:
                trace.marker.size = marker_size
                trace.marker.line = dict(color=line_color, width=1)
            else:
                trace.marker.size = marker_size

    # Tick font size
    tick_font_size = int(font_size * 6 / 7)
    axis_format = dict(
        showbackground=False,
        visible=not hide_axes,
        showgrid=show_gridlines,
        zeroline=True,
        gridcolor=line_color,
        linecolor=line_color,
        zerolinecolor=line_color,
        tickfont=dict(size=tick_font_size)
    )
    # Create a copy for each axis and assign titles
    dict_list = []
    for i, label in enumerate(axis_labels):
        d = axis_format.copy()
        d['title'] = label
        dict_list.append(d)

    # Handle axis ranges and aspect ratio
    scene_kwargs = {}
    x_range = fig.layout.scene.xaxis.range
    y_range = fig.layout.scene.yaxis.range
    z_range = fig.layout.scene.zaxis.range
    if x_range is not None:
        x_ext = x_range[1] - x_range[0]
        y_ext = y_range[1] - y_range[0]
        z_ext = z_range[1] - z_range[0]
        max_ext = max(x_ext, y_ext, z_ext, 1e-9)
        ar = dict(x=x_ext/max_ext, y=y_ext/max_ext, z=z_ext/max_ext)
        dict_list[0]['range'] = x_range
        dict_list[1]['range'] = y_range
        dict_list[2]['range'] = z_range
        scene_kwargs['aspectmode'] = 'manual'
        scene_kwargs['aspectratio'] = ar
    else:
        scene_kwargs['aspectmode'] = aspectmode
    scene_kwargs['xaxis'] = dict_list[0]
    scene_kwargs['yaxis'] = dict_list[1]
    scene_kwargs['zaxis'] = dict_list[2]

    # Apply scene settings
    fig.update_layout(scene=scene_kwargs)

    # General layout
    fig.update_layout(
        font=dict(color=text_color, family="Arial, sans-serif", size=font_size),
        plot_bgcolor=bk_color,
        paper_bgcolor=bk_color,
        scene_camera=dict(eye=eye),
    )
    # Colorbar positioning if legends present
    if any(getattr(trace, 'showlegend', True) for trace in fig.data if hasattr(trace, 'marker')):
        fig.update_layout(coloraxis_colorbar=dict(x=1, y=0, len=0.5, yanchor='bottom'))
    else:
        fig.update_layout(width=dims[1], height=dims[0], title=title,
                          coloraxis_colorbar=dict(x=1, y=0.5, len=0.5))

    return fig



def rotate_figure(fig, zoom_factor=1.0, z_rotation=0, elev_rotation=0):
    """
    Adjust the camera perspective of a Plotly 3D figure.

    Parameters:
      fig (go.Figure): Plotly figure object with a 3D scene.
      zoom_factor (float): Multiplicative factor to scale the camera's distance from the center.
                           Values < 1 zoom in; values > 1 zoom out.
      z_rotation (float): Rotation (in degrees) about the z-axis (i.e. in the x-y plane).
      elev_rotation (float): Rotation (in degrees) to change the elevation (i.e. the polar angle).
                             Positive values will tilt the camera; negative values will lower it.

    Returns:
      The updated Plotly figure.
    """

    # Get current camera eye position; if not set, use a default.
    try:
        current_eye = fig.layout.scene.camera.eye
        # Assume current_eye is a dict with keys "x", "y", and "z".
        x = current_eye.get("x", 1.25)
        y = current_eye.get("y", 1.25)
        z = current_eye.get("z", 1.25)
    except Exception:
        x, y, z = 1.25, 1.25, 1.25

    # --- 1) original camera vector
    #    (x,y,z) is your input eye position before zooming
    orig_r = math.sqrt(x * x + y * y + z * z)
    if orig_r == 0:
        raise ValueError("Original eye at the origin is invalid")

    # --- 2) decide your zoomed radius
    zoomed_r = orig_r / zoom_factor

    # --- 3) get the original direction unit‐vector angles
    dx, dy, dz = x / orig_r, y / orig_r, z / orig_r
    theta = math.atan2(dy, dx)  # azimuth in [–π,π]
    phi = math.acos(max(-1, min(1, dz)))  # polar angle in [0,π]

    # --- 4) apply your desired rotations
    theta += math.radians(z_rotation)  # spin around z
    phi += math.radians(elev_rotation)
    phi = max(0, min(math.pi, phi))  # clamp to [0,π]

    # --- 5) reconstruct with the *fixed* zoomed radius
    new_x = zoomed_r * math.sin(phi) * math.cos(theta)
    new_y = zoomed_r * math.sin(phi) * math.sin(theta)
    new_z = zoomed_r * math.cos(phi)

    # --- 6) update Plotly’s camera
    new_camera = {"eye": {"x": new_x, "y": new_y, "z": new_z}}
    fig.update_layout(scene=dict(camera=new_camera))

    return fig

