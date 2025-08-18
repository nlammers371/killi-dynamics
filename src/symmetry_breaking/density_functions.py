import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def compute_spherical_knn_density(
    tracks_df: pd.DataFrame,
    sphere_df: pd.DataFrame,
    k: int = 10,
    method: str = "trimmed",   # "kth" | "avg" | "trimmed"
    alpha: float = 0.8,        # used only for "trimmed": q = ceil(alpha*k)
    project_to_sphere: bool = True,
    time_col: str = "t",
    xcol: str = "x", ycol: str = "y", zcol: str = "z",
    sphere_time_col: str = "t",
    sphere_center_cols = ("xc","yc","zc"),
    sphere_radius_col: str = "r",
    return_extras: bool = False
):
    """
    For each timepoint, estimate local surface density at each cell using KNN on a sphere.

    Parameters
    ----------
    tracks_df : DataFrame with columns [t, x, y, z, ...]
    sphere_df : DataFrame with columns [t, xc, yc, zc, r]
    k : number of neighbors (excluding self)
    method : "kth" (K/A(theta_K)), "avg" (mean over j/A(theta_j)), "trimmed" (q/A(theta_q))
    alpha : trimming fraction for "trimmed" (0<alpha<=1); q = ceil(alpha*k)
    project_to_sphere : normalize each point radially to given sphere before KNN
    return_extras : if True, also return kth angles and densities per-j

    Returns
    -------
    out_df : tracks_df with new columns:
        - 'rho_sphere': local surface density (cells / unit area)
        - optionally: 'theta_k', 'rho_per_j' (array) if return_extras=True
    """
    assert 0 < k, "k must be positive"
    if method not in {"kth", "avg", "trimmed"}:
        raise ValueError("method must be 'kth', 'avg', or 'trimmed'")
    if method == "trimmed":
        q = int(np.ceil(alpha * k))
        q = max(1, min(q, k))

    # Join sphere params by time for quick lookup
    sph = sphere_df.set_index(sphere_time_col)[list(sphere_center_cols)+[sphere_radius_col]]

    # Prepare outputs
    rho_list = []
    theta_k_list = [] if return_extras else None
    rho_per_j_list = [] if return_extras else None

    # We will fill in order of tracks_df; collect per-timepoint then concatenate
    out_indices = []
    for t_val, df_t in tracks_df.groupby(time_col, sort=False):
        if t_val not in sph.index:
            raise KeyError(f"Sphere parameters missing for t={t_val}")
        xc, yc, zc = sph.loc[t_val, list(sphere_center_cols)].values
        r = float(sph.loc[t_val, sphere_radius_col])

        pts = df_t[[xcol, ycol, zcol]].to_numpy(float)
        # Translate to sphere-centered coords
        v = pts - np.array([xc, yc, zc], dtype=float)

        # Radial projection to sphere (unit vectors u, and optionally new 3D positions)
        norms = np.linalg.norm(v, axis=1)
        # Guard against zeros
        norms = np.where(norms == 0, 1e-12, norms)
        u = v / norms[:, None]  # unit vectors
        if project_to_sphere:
            # projected positions (not strictly needed for KNN since we use u)
            _proj = np.array([xc, yc, zc]) + r * u

        # Build KDTree on unit sphere vectors; chord distance is fine
        tree = cKDTree(u)
        # query k+1 because the nearest neighbor is the point itself (distance 0)
        dists, idxs = tree.query(u, k=k+1, n_jobs=-1)  # chord distances on unit sphere
        # drop self (column 0)
        chord = dists[:, 1:]   # shape (N_t, k)
        # convert chord -> angle
        # numerical guard: clip to [0, 2]
        chord = np.clip(chord, 0.0, 2.0)
        theta = 2.0 * np.arcsin(0.5 * chord)  # radians on unit sphere
        # spherical-cap areas on radius r
        A = 2.0 * np.pi * (r**2) * (1.0 - np.cos(theta))  # (N_t, k)

        # densities per j (using j neighbors inside A(theta_j))
        js = np.arange(1, k+1, dtype=float)[None, :]  # shape (1, k)
        rho_per_j = js / A  # (N_t, k)

        if method == "kth":
            rho = rho_per_j[:, -1]  # K/A(theta_K)
            theta_k = theta[:, -1]
        elif method == "avg":
            rho = rho_per_j.mean(axis=1)  # average over j
            theta_k = theta[:, -1]
        else:  # "trimmed"
            rho = rho_per_j[:, q-1]       # q/A(theta_q)
            theta_k = theta[:, q-1]

        rho_list.append(rho)
        if return_extras:
            theta_k_list.append(theta_k)
            rho_per_j_list.append(rho_per_j)
        out_indices.append(df_t.index.values)

    # Stitch back together respecting original row order
    rho_all = np.concatenate(rho_list, axis=0)
    idx_all = np.concatenate(out_indices, axis=0)

    out_df = tracks_df.copy()
    out_df.loc[idx_all, "rho_sphere"] = rho_all

    if return_extras:
        out_df.loc[idx_all, "theta_k"] = np.concatenate(theta_k_list, axis=0)
        # store rho_per_j as object arrays (one small array per row)
        rpj = np.concatenate(rho_per_j_list, axis=0)
        out_df.loc[idx_all, "rho_per_j"] = list(rpj)

    return out_df
