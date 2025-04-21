import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import distance_matrix
from scipy.special import sph_harm
from scipy.spatial.transform import Rotation as R


def build_sh_basis(L_max, phi, theta):
    basis_functions = []
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            # Note: sph_harm takes (m, l, phi, theta)
            Y_lm = sph_harm(m, l, phi, theta)
            basis_functions.append(Y_lm.real)

    return np.column_stack(basis_functions).T

# Assuming vertices is an (N, 3) array and radial_distances is the corresponding data.
def cart2sph(points_xyz, v=None):
    # points_xyz: (N,3) array of [x,y,z]
    if v is None:
        v = np.array([0, 0, 1], float)
    else:
        v = np.asarray(v, float)

    # 1) normalize v
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        raise ValueError("Up‐vector v must be nonzero")
    v_unit = v / norm_v

    # 2) compute the rotation that sends v->target
    target = np.array([0, 0, 1], float)  # new “north” axis
    # align_vectors(A, B) finds R so that R @ A[i] == B[i]
    rot, _ = R.align_vectors([v_unit], [target])
    Rmat = rot.as_matrix()

    # 3) apply the rotation to all points
    pts_rot = points_xyz @ Rmat.T

    # 4) convert to spherical
    x, y, z = pts_rot.T
    r = np.linalg.norm(pts_rot, axis=1)
    # guard against zero‐length
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.arccos(np.clip(z / r, -1.0, 1.0))  # 0…π
    phi = np.arctan2(y, x)                          # –π…π

    return np.column_stack([r, theta, phi])

def sph2cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def create_sphere_mesh(center, radius, resolution=50):
    # Create a grid of angles
    phi, theta = np.mgrid[0.0:np.pi:complex(0, resolution), 0.0:2.0 * np.pi:complex(0, resolution)]

    # Parametric equations for a sphere
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)

    # Create vertices array
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    # Create faces: two triangles for each square in the grid
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx0 = i * resolution + j
            idx1 = idx0 + 1
            idx2 = idx0 + resolution
            idx3 = idx2 + 1
            faces.append([idx0, idx2, idx1])
            faces.append([idx1, idx2, idx3])
    faces = np.array(faces)
    return vertices, faces

def fit_sphere(points, quantile=0.95):
    """
    Fits a sphere to the given 3D points.

    Parameters:
        points (np.ndarray): Nx3 array of 3D coordinates.
        mode (str): 'inner', 'outer', or 'average'.
        quantile (float): quantile for inner/outer fitting (0-1).

    Returns:
        center (np.ndarray): sphere center coordinates.
        radius (float): radius of the fitted sphere.
    """
    # Initial guess for sphere center: centroid of points
    center_init = np.mean(points, axis=0)

    def residuals(c):
        r = np.linalg.norm(points - c, axis=1)
        # if mode == 'average':
        return r - r.mean()


    result = least_squares(residuals, center_init)
    fitted_center = result.x

    # Compute final radius based on chosen mode
    final_distances = np.linalg.norm(points - fitted_center, axis=1)
    fitted_radius = final_distances.mean()
    outer_radius = np.quantile(final_distances, 1 - quantile)
    inner_radius = np.quantile(final_distances, quantile)

    return fitted_center, fitted_radius, inner_radius, outer_radius

def create_sh_mesh(coeffs, sphere_mesh):
    """
    Evaluates the spherical harmonics on the sphere mesh.

    Parameters:
        coeffs (np.ndarray): coefficients of the spherical harmonics.
        sphere_mesh (tuple): vertices and faces of the sphere mesh.
        radius (float): radius of the sphere.

    Returns:
        np.ndarray: evaluated values on the sphere mesh.
    """

    vertices, _ = sphere_mesh

    mesh_center = np.mean(vertices, axis=0)  # center of the sphere mesh
    vertices_c = vertices - mesh_center  # shift vertices to center

    # Convert Cartesian coordinates to spherical coordinates
    r, theta, phi = cart2sph(vertices_c[:, 0], vertices_c[:, 1], vertices_c[:, 2])

    L_max = int(np.sqrt(len(coeffs)) - 1)
    # Build the basis functions
    basis_functions = build_sh_basis(L_max, phi=phi, theta=theta)

    # Evaluate the spherical harmonics
    r_sh = coeffs[None, :] @ basis_functions

    # get new caresian points
    x, y, z = sph2cart(r_sh, theta, phi)
    vertices_sh = np.c_[x.T, y.T, z.T] + mesh_center  # combine x, y, z into a single array

    # define SH mesh
    sh_mesh = (vertices_sh, sphere_mesh[1])  # keep the same faces as the original sphere mesh
    # sh_mesh.vertices = vertices_sh  # update vertices with SH values

    return sh_mesh, r_sh


# write function to fit spherical harmoncs to deviations from sphere surface
def fit_sphere_and_sh(points, L_max=10, knn=3, k_thresh=50, sphere_quantile=0.25):
    """
    Fits spherical harmonics to the deviations of points from a sphere.

    Parameters:
        points (np.ndarray): Nx3 array of 3D coordinates.
        radius (float): radius of the sphere.
        order (int): maximum order of spherical harmonics.

    Returns:
        coeffs (np.ndarray): coefficients of the fitted spherical harmonics.
    """
    # first, fit a sphere to the points
    fitted_center, fitted_radius, inner_radius, outer_radius = fit_sphere(points, quantile=sphere_quantile)
    scale_factor = inner_radius / fitted_radius
    # shift points to center
    points_c = points - fitted_center

    # Convert Cartesian coordinates to spherical coordinates
    r, theta, phi = cart2sph(points_c[:, 0], points_c[:, 1], points_c[:, 2])

    # Generate the mesh for the sphere:
    vertices, faces = create_sphere_mesh(np.asarray([0, 0, 0]), fitted_radius, resolution=100)
    r_v, theta_v, phi_v = cart2sph(vertices[:, 0], vertices[:, 1], vertices[:, 2])

    # map centroids to sphere vertices
    surf_dist_mat = distance_matrix(vertices, points_c)
    closest_indices = np.argsort(surf_dist_mat, axis=1)[:, :knn]
    closest_distances = np.sort(surf_dist_mat, axis=1)[:, :knn]
    r_dist_array = r[closest_indices]
    r_dist_array[closest_distances > k_thresh] = np.nan
    radial_distances = np.nanmean(r_dist_array, axis=1)

    radial_distances[np.isnan(radial_distances)] = fitted_radius
    radial_distances = radial_distances * scale_factor

    # Compute spherical harmonics
    basis_functions = build_sh_basis(L_max, theta=theta_v, phi=phi_v)

    # Create the design matrix (N, num_basis)
    Y = np.column_stack(basis_functions)

    # nan_filter = ~np.isnan(radial_distances)
    Y_filtered = Y
    radial_distances_filtered = radial_distances

    # Solve for coefficients using least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(Y_filtered, radial_distances_filtered, rcond=None)

    # Evaluate the fitted function at the vertices
    # fitted_radial = Y @ coeffs

    return np.array(coeffs), fitted_center, inner_radius, radial_distances