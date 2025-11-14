import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
import numpy as np
from sklearn.linear_model import LinearRegression

def smooth_tracks_savgol(df: pd.DataFrame,
                         window: int = 5,
                         poly: int = 2,
                         coords=('x', 'y', 'z')):
    """
    Apply Savitzky–Golay smoothing to xyz coords per track.
    Assumes df has columns: track_id, t, and coords.
    Returns a copy with new columns: x_smooth, y_smooth, z_smooth.
    """
    out = df.copy()
    out = out.sort_values(['track_id', 't'])

    for c in coords:
        out[f'{c}_smooth'] = None

    for tid, sub in out.groupby('track_id', group_keys=False):
        # ensure numeric continuity
        sub = sub.sort_values('t')

        if len(sub) < window:
            # not enough points → no smoothing, just copy
            for c in coords:
                out.loc[sub.index, f'{c}_smooth'] = sub[c].values
            continue

        for c in coords:
            sm = savgol_filter(sub[c].values, window_length=window, polyorder=poly, mode='interp')
            out.loc[sub.index, f'{c}_smooth'] = sm

    return out



def geodesic_distances_on_sphere(tracks_df, wound_df, ref_vec=np.array([0., 0., 1.]), wrap_phi=True):
    """
    tracks_df: rows = multiple points per frame, cols ["t","x","y","z"]
    wound_df:  one row per frame with cols ["t","xw","yw","zw"]
    sphere_df: one row per frame with cols ["t","xs","ys","zs","r"] (center & radius)

    Returns: tracks_df merged with wound/sphere info + added columns:
        - "theta": central angle (radians) between wound and point (same as before)
        - "d_geo": great-circle distance along the sphere (= r * theta)
        - "d_chord": straight-line Euclidean distance (for reference)
        - "phi": azimuthal angle around the wound center (radians), measured in the tangent plane
                 relative to a projected reference direction (ref_vec), increasing via right-hand rule.
    Notes:
        - (theta, phi) give you local polar coords on the sphere around the wound.
        - ref_vec sets where phi=0 points after projection into the tangent plane at the wound.
    """
    # Merge wound & sphere info onto each track row by time
    df = tracks_df.merge(wound_df, on="t", how="inner")#.merge(sphere_df, on="t", how="inner")

    # Extract arrays
    P = df[["x","y","z"]].to_numpy(dtype=float)           # track points
    W = df[["xw","yw","zw"]].to_numpy(dtype=float)        # wound points
    C = df[["xs","ys","zs"]].to_numpy(dtype=float)        # sphere centers
    r = df["r"].to_numpy(dtype=float)                     # radii

    # Unit vectors from center to each point (guard against division by zero)
    U = (P - C) / r[:, None]   # direction to track point
    N = (W - C) / r[:, None]   # direction to wound point ("Up" at wound)

    # Normalize (handles small numerical drift)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    N /= np.linalg.norm(N, axis=1, keepdims=True)

    # --- Central angle (theta) and great-circle distance ---
    dot = np.einsum("ij,ij->i", U, N)
    dot = np.clip(dot, -1.0, 1.0)
    cross_norm = np.linalg.norm(np.cross(U, N), axis=1)
    theta = np.arctan2(cross_norm, dot)              # radians
    d_geo = r * theta
    d_chord = np.linalg.norm(P - W, axis=1)

    # --- Local tangent frame at wound point to get azimuth phi ---
    # Project ref_vec into each tangent plane at N to define phi=0 direction
    ref = np.asarray(ref_vec, dtype=float)
    if np.linalg.norm(ref) == 0:
        raise ValueError("ref_vec must be non-zero.")

    # For each row, project ref into tangent plane at N[i]: e0 = normalize(ref - (ref·N)N)
    ref_dot_N = np.einsum("i,ij->j", ref, N.T)  # shape (M,) via broadcasting trick
    # But that trick is awkward; do explicit:
    ref_dot_N = (N @ ref)                       # (M,)
    e0 = ref - ref_dot_N[:, None] * N           # remove normal component
    e0_norm = np.linalg.norm(e0, axis=1, keepdims=True)

    # Fallback if ref is (nearly) parallel to N: use global x-axis as backup
    bad = (e0_norm[:, 0] < 1e-12)
    if np.any(bad):
        fallback = np.array([1.0, 0.0, 0.0])
        ref_dot_N_fb = (N[bad] @ fallback)
        e0_fb = fallback - ref_dot_N_fb[:, None] * N[bad]
        e0[bad] = e0_fb
        e0_norm[bad] = np.linalg.norm(e0_fb, axis=1, keepdims=True)

    e0 = e0 / e0_norm                             # phi = 0 direction in tangent plane
    e1 = np.cross(N, e0)                          # 90° CCW about N to complete right-handed basis

    # Tangent component of U relative to N
    U_par = np.einsum("ij,ij->i", U, N)           # scalar component along N
    U_tan = U - U_par[:, None] * N
    # Normalize tangent direction to get pure direction on tangent plane
    U_tan_norm = np.linalg.norm(U_tan, axis=1, keepdims=True)
    # If point coincides with N (theta≈0), U_tan is ~0; define phi=0 there
    near = (U_tan_norm[:, 0] < 1e-12)
    U_tan_dir = np.zeros_like(U_tan)
    U_tan_dir[~near] = U_tan[~near] / U_tan_norm[~near]

    # Azimuth phi via projections onto (e0, e1)
    X = P[:, 0] - W[:, 0] # np.einsum("ij,ij->i", U_tan_dir, e0)
    Y = P[:, 1] - W[:, 1] # np.einsum("ij,ij->i", U_tan_dir, e1)
    phi = np.mod(np.arctan2(X, Y), 2*np.pi)                     # (-pi, pi]

    if wrap_phi:
        phi = (phi + 2*np.pi) % (2*np.pi)         # [0, 2*pi)

    out = df.copy()
    out["theta"] = theta
    out["d_geo"] = d_geo
    out["d_chord"] = d_chord
    out["phi"] = phi
    return out

def compute_windowed_msd(positions, W=40, step=5,
                         max_tau=10,
                         fit_tau_min=2, fit_tau_max=6):
    """
    positions: (N,3) array for a single track
    Returns list of dicts, one per window:
        {
            'start_idx': i,
            'end_idx': i+W,
            'taus': array([...]),
            'msd': array([...]),
            'alpha': slope or np.nan,
        }
    """

    N = len(positions)
    results = []

    for i in range(0, N - W + 1, step):
        seg = positions[i:i+W]
        displacements = seg[None, :, :] - seg[:, None, :]
        # displacement[t, t+tau, :] gives vector difference

        taus = np.arange(1, max_tau+1)
        msd_vals = []

        for tau in taus:
            if tau >= W:
                msd_vals.append(np.nan)
                continue
            diffs = seg[tau:] - seg[:-tau]
            msd_vals.append(np.mean(np.sum(diffs**2, axis=1)))

        msd_vals = np.array(msd_vals)

        # Fit α: log(MSD) = α log(tau) + const
        # Use only good entries
        fit_mask = (
                (taus >= fit_tau_min) &
                (taus <= fit_tau_max) &
                np.isfinite(msd_vals) &
                (msd_vals > 0)  # exclude zeros (log(0)=-inf)
        )
        if np.sum(fit_mask) >= 3:
            X = np.log(taus[fit_mask]).reshape(-1, 1)
            y = np.log(msd_vals[fit_mask])
            lr = LinearRegression().fit(X, y)
            alpha = lr.coef_[0]
        else:
            alpha = np.nan

        results.append({
            'start_idx': i,
            'end_idx': i+W,
            'taus': taus,
            'msd': msd_vals,
            'alpha': alpha
        })

    return results


def compute_windowed_autocorr(positions, W=20, step=5):
    """
    positions: (N,3) array for one track
    Returns list of dicts:
        {
            'start_idx': i,
            'end_idx': i+W,
            'C1': autocorrelation at lag-1,
            'mean_cos_theta': directional coherence measure
        }
    """

    N = len(positions)
    results = []

    # displacement vectors
    v = positions[1:] - positions[:-1]   # shape (N-1, 3)

    # normalize for turning angles
    v_norm = np.linalg.norm(v, axis=1)
    valid = v_norm > 0
    v_unit = np.zeros_like(v)
    v_unit[valid] = v[valid] / v_norm[valid, None]

    for i in range(0, N - W, step):
        seg_v = v[i:i+W-1]          # includes W-1 velocity vectors
        seg_unit = v_unit[i:i+W-1]
        seg_norm = v_norm[i:i+W-1]

        if len(seg_v) < 2:
            results.append({'start_idx': i,
                            'end_idx': i+W,
                            'C1': np.nan,
                            'mean_cos_theta': np.nan})
            continue

        # Autocorrelation lag 1:
        # C(1) = <v_t · v_{t+1}> / <|v_t|^2>
        dot = np.sum(seg_v[:-1] * seg_v[1:], axis=1)
        denom = np.sum(seg_norm[:-1]**2)
        C1 = np.sum(dot) / denom if denom > 0 else np.nan

        # Turning-angle coherence: mean cos(theta)
        # cos(theta_t) = v_hat_t · v_hat_{t+1}
        valid_u = (np.linalg.norm(seg_unit[:-1],axis=1)>0) & \
                  (np.linalg.norm(seg_unit[1:],axis=1)>0)
        if np.any(valid_u):
            cos_t = np.sum(seg_unit[:-1][valid_u] * seg_unit[1:][valid_u], axis=1)
            mean_cos = np.mean(cos_t)
        else:
            mean_cos = np.nan

        results.append({
            'start_idx': i,
            'end_idx': i+W,
            'C1': C1,
            'mean_cos_theta': mean_cos
        })

    return results

