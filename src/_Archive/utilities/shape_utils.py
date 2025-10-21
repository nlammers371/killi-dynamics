import plotly.graph_objects as go


def calculate_face_centroids(mesh):

    f = mesh.faces
    v = mesh.vertices

    face_vertices = v[f]

    # Compute the centroids of each face
    face_centroids = face_vertices.mean(axis=1)

    return face_centroids

def plot_mesh(plot_hull, surf_alpha=0.2):
    tri_points = plot_hull.vertices[plot_hull.faces]

    # extract the lists of x, y, z coordinates of the triangle vertices and connect them by a line
    Xe = []
    Ye = []
    Ze = []
    for T in tri_points:
        Xe.extend([T[k % 3][0] for k in range(4)] + [None])
        Ye.extend([T[k % 3][1] for k in range(4)] + [None])
        Ze.extend([T[k % 3][2] for k in range(4)] + [None])

    # define the trace for triangle sides
    fig = go.Figure()
    lines = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode='lines',
        name='',
        line=dict(color='rgb(70,70,70, 0.5)', width=1))

    lighting_effects = dict(ambient=0.4, diffuse=0.5, roughness=0.9, specular=0.9, fresnel=0.9)
    mesh = go.Mesh3d(x=plot_hull.vertices[:, 0], y=plot_hull.vertices[:, 1], z=plot_hull.vertices[:, 2],
                     opacity=surf_alpha, i=plot_hull.faces[:, 0], j=plot_hull.faces[:, 1], k=plot_hull.faces[:, 2],
                     lighting=lighting_effects)
    fig.add_trace(mesh)

    fig.add_trace(lines)
    fig.update_layout(template="plotly")

    return fig, lines, mesh

#
# def fit_fin_mesh(xyz_fin, alpha=20, n_faces=5000, smoothing_strength=5):
#     # normalize for alphshape fitting
#     mp = np.min(xyz_fin)
#     points = xyz_fin - mp
#     mmp = np.max(points)
#     points = points / mmp
#
#     meshing_error_flag = False
#     try:
#         raw_hull = alphashape.alphashape(points, alpha)
#     except:
#         meshing_error_flag = True
#
#     if not meshing_error_flag:
#         # copy
#         hull02_cc = raw_hull.copy()
#
#         # keep only largest component
#         hull02_cc = hull02_cc.split(only_watertight=False)
#         hull02_sm = max(hull02_cc, key=lambda m: m.area)
#
#         hull02_sm = trimesh.smoothing.filter_laplacian(hull02_sm, iterations=2)
#
#         # fill holes
#         hull02_sm = mesh_cleanup(hull02_sm)
#
#         # smooth
#         hull02_sm = trimesh.smoothing.filter_laplacian(hull02_sm, iterations=smoothing_strength)
#
#         # resample
#         n_faces = np.min([n_faces, hull02_sm.faces.shape[0] - 1])
#         hull02_rs = hull02_sm.simplify_quadric_decimation(face_count=n_faces)
#         hull02_rs = hull02_rs.split(only_watertight=False)
#         hull02_rs = max(hull02_rs, key=lambda m: m.area)
#         hull02_rs.fill_holes()
#         hull02_rs.fix_normals()
#
#         vt = hull02_rs.vertices
#         vt = vt * mmp
#         vt = vt + mp
#         hull02_rs.vertices = vt
#
#         # check
#         wt_flag = hull02_rs.is_watertight
#
#         return hull02_rs, raw_hull, wt_flag
#
#     else:
#         return None, None, False