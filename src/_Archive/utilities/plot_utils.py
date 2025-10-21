import plotly.graph_objects as go


def mesh_face_plot(faces, vertices, intensity, colormap="Viridis"):

    # Duplicate vertices to make each face unique
    x_faces = []
    y_faces = []
    z_faces = []
    intensity_faces = []

    for face, intensity in zip(faces, intensity):
        for vertex_idx in face:  # Add each vertex of the face
            x_faces.append(vertices[vertex_idx][0])
            y_faces.append(vertices[vertex_idx][1])
            z_faces.append(vertices[vertex_idx][2])
        intensity_faces.extend([intensity] * 3)   # Same intensity for all vertices of the face

    # Build new indices for the triangular faces
    n_faces = len(faces)
    i = [3 * j for j in range(n_faces)]
    j = [3 * j + 1 for j in range(n_faces)]
    k = [3 * j + 2 for j in range(n_faces)]
    # Create the Plotly Mesh3d plot

    # Create the Plotly Mesh3d plot
    fig = go.Figure(data=[go.Mesh3d(
        x=x_faces,
        y=y_faces,
        z=z_faces,
        i=i,
        j=j,
        k=k,
        intensity=intensity_faces,
        colorscale=colormap,
        flatshading=True
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    return fig