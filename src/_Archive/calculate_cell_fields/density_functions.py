import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def xcorr_laglead_by_track(
    df: pd.DataFrame,
    var_x: str,
    var_y: str,
    min_t: int = 10,
    max_lag: int = 20,
    time_col: str = "t",
    id_col: str = "track_id",
    dropna: bool = True
):
    """
    Compute normalized cross-correlation rho_xy(τ) for τ in [-max_lag, ..., +max_lag]
    for each track_id. Uses per-lag normalization by (n - |τ|)*std(x)*std(y) to keep
    correlations within [-1, 1] (except degenerate std=0 -> NaN).

    Returns
    -------
    results : list of dicts with keys:
        - 'track_id'
        - 'lags'  (np.ndarray, shape (2*max_lag+1,))
        - 'xcorr' (np.ndarray, same shape, dtype float)
        - 'n'     (int, length of the track after any NA handling)
    """
    results = []
    lags = np.arange(-max_lag, max_lag + 1, dtype=int)

    for tid, dft in df.groupby(id_col):
        dft = dft.sort_values(time_col)

        x = dft[var_x].to_numpy(dtype=float)
        y = dft[var_y].to_numpy(dtype=float)

        if dropna:
            # keep only rows where both present
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]

        n = len(x)
        if n < max(min_t, 2):  # need at least 2 points to define std
            continue

        xmu, ymu = x.mean(), y.mean()
        xs, ys = x.std(ddof=0), y.std(ddof=0)
        if xs == 0 or ys == 0:
            # degenerate: correlation undefined
            results.append({"track_id": tid, "lags": lags, "xcorr": np.full_like(lags, np.nan, dtype=float), "n": n})
            continue

        x_cent = x - xmu
        y_cent = y - ymu

        corr = np.empty_like(lags, dtype=float)
        for i, tau in enumerate(lags):
            if tau >= 0:
                # correlate x_t with y_{t+tau}
                m = n - tau
                num = np.dot(x_cent[:m], y_cent[tau:])
            else:
                # tau < 0: correlate x_{t - tau} with y_t  (equivalently y leads)
                m = n + tau  # since tau negative
                num = np.dot(x_cent[-tau:], y_cent[:m])
            denom = m * xs * ys
            corr[i] = num / denom if denom != 0 else np.nan

        results.append({"track_id": tid, "lags": lags, "xcorr": corr, "n": n})

    return results




def compute_spherical_knn_density(
    tracks_df: pd.DataFrame,
    sphere_df: pd.DataFrame,
    k: int = 10,
    method: str = "trimmed",   # "kth" | "avg" | "trimmed"
    alpha: float = 0.8,        # used only for "trimmed": q = ceil(alpha*k)
    project_to_sphere: bool = True,
    # column names
    time_col: str = "t",
    xcol: str = "x", ycol: str = "y", zcol: str = "z",
    sphere_time_col: str = "t",
    sphere_center_cols = ("xc","yc","zc"),
    sphere_radius_col: str = "r",
    fluo_col: str = "mean_fluo",
    # fluorescence aggregation behavior
    fluo_include_self: bool = True,   # include the reference cell in the fluo aggregation
    fluo_k: int | None = None,        # if None, use k; otherwise use this many neighbors for fluo
    return_extras: bool = False
):
    """
    Estimate per-cell spherical KNN density and aggregate fluorescence over neighbors.

    Adds columns:
      - 'rho_sphere'      : local surface density (cells / unit area) using K neighbors (excluding self)
      - 'fluo_mean_knn'   : mean of `fluo_col` over neighbors (and optionally self)
      - 'fluo_sum_knn'    : sum of `fluo_col` over neighbors (and optionally self)
      - optionally 'theta_k', 'rho_per_j' if return_extras=True

    Notes:
      * Density uses the K nearest neighbors EXCLUDING self (standard).
      * Fluorescence uses `fluo_k` neighbors (default = K), and includes self if `fluo_include_self=True`.
    """
    assert 0 < k, "k must be positive"
    if method not in {"kth", "avg", "trimmed"}:
        raise ValueError("method must be 'kth', 'avg', or 'trimmed'")
    if method == "trimmed":
        q = int(np.ceil(alpha * k))
        q = max(1, min(q, k))

    fluo_k = k if fluo_k is None else int(fluo_k)
    if fluo_k <= 0:
        raise ValueError("fluo_k must be positive")

    # Join sphere params by time
    sph = sphere_df.set_index(sphere_time_col)[list(sphere_center_cols)+[sphere_radius_col]]

    rho_list, fluo_mean_list, fluo_sum_list = [], [], []
    theta_k_list = [] if return_extras else None
    rho_per_j_list = [] if return_extras else None
    out_indices = []

    for t_val, df_t in tracks_df.groupby(time_col, sort=False):
        if t_val not in sph.index:
            raise KeyError(f"Sphere parameters missing for t={t_val}")
        xc, yc, zc = sph.loc[t_val, list(sphere_center_cols)].values
        r = float(sph.loc[t_val, sphere_radius_col])

        pts = df_t[[xcol, ycol, zcol]].to_numpy(float)
        v = pts - np.array([xc, yc, zc], dtype=float)

        norms = np.linalg.norm(v, axis=1)
        norms = np.where(norms == 0, 1e-12, norms)
        u = v / norms[:, None]  # unit vectors on unit sphere

        # KDTree on unit vectors (chord distance ~ angular distance)
        tree = cKDTree(u)

        # --- Density neighbors (exclude self) ---
        dists_k, idxs_k = tree.query(u, k=k+1)  # first is self
        chord = np.clip(dists_k[:, 1:], 0.0, 2.0)          # drop self for density
        theta = 2.0 * np.arcsin(0.5 * chord)               # radians
        A = 2.0 * np.pi * (r**2) * (1.0 - np.cos(theta))   # spherical cap areas
        js = np.arange(1, k+1, dtype=float)[None, :]
        rho_per_j = js / A

        if method == "kth":
            rho = rho_per_j[:, -1]
            theta_k = theta[:, -1]
        elif method == "avg":
            rho = rho_per_j.mean(axis=1)
            theta_k = theta[:, -1]
        else:  # trimmed
            rho = rho_per_j[:, q-1]
            theta_k = theta[:, q-1]

        # --- Fluorescence aggregation ---
        # query fluo_k neighbors; include self by using fluo_k if include_self,
        # or fluo_k+1 if we need to exclude self from the neighbor set
        query_k_f = fluo_k if fluo_include_self else (fluo_k + 1)
        dists_f, idxs_f = tree.query(u, k=query_k_f)

        if fluo_include_self:
            # `idxs_f` already includes self; use as is
            neigh_indices = idxs_f
        else:
            # drop the self column (index where distance==0): typically column 0
            # robustly drop by checking the smallest distance per row
            # (works even if duplicates collapse to same point)
            self_pos = np.argmin(dists_f, axis=1)[:, None]
            mask = np.ones_like(idxs_f, dtype=bool)
            # zero out the self positions
            np.put_along_axis(mask, self_pos, False, axis=1)
            # keep the closest `fluo_k` *excluding* self
            # to do that, we sort by distance per row and take first fluo_k
            order = np.argsort(dists_f, axis=1)
            top = order[:, :fluo_k]
            neigh_indices = np.take_along_axis(idxs_f, top, axis=1)

        fluo_vals = df_t[fluo_col].to_numpy()
        # handle NaNs gracefully: mean over finite values; if all NaN -> NaN
        neigh_fluo = fluo_vals[neigh_indices]  # shape (N_t, fluo_k or fluo_k including self)
        with np.errstate(invalid="ignore"):
            fluo_mean = np.nanmean(neigh_fluo, axis=1)
            fluo_sum  = np.nansum(neigh_fluo, axis=1)

        # collect
        rho_list.append(rho)
        fluo_mean_list.append(fluo_mean)
        fluo_sum_list.append(fluo_sum)
        if return_extras:
            theta_k_list.append(theta_k)
            rho_per_j_list.append(rho_per_j)
        out_indices.append(df_t.index.values)

    # Stitch back
    rho_all = np.concatenate(rho_list, axis=0)
    fluo_mean_all = np.concatenate(fluo_mean_list, axis=0)
    fluo_sum_all = np.concatenate(fluo_sum_list, axis=0)
    idx_all = np.concatenate(out_indices, axis=0)

    out_df = tracks_df.copy()
    out_df.loc[idx_all, "rho_sphere"] = rho_all
    out_df.loc[idx_all, "fluo_mean_knn"] = fluo_mean_all
    out_df.loc[idx_all, "fluo_sum_knn"] = fluo_sum_all

    if return_extras:
        out_df.loc[idx_all, "theta_k"] = np.concatenate(theta_k_list, axis=0)
        rpj = np.concatenate(rho_per_j_list, axis=0)
        out_df.loc[idx_all, "rho_per_j"] = list(rpj)

    return out_df
