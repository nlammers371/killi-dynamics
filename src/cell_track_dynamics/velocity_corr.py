import numpy as np
from sklearn.neighbors import BallTree
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
import pandas as pd

# ------------------------------------------------------------
# WINDOWS-SAFE wrapper for process_map
# ------------------------------------------------------------
def _unwrap_helper(func, subdf):
    return func(subdf)


# ------------------------------------------------------------
# Worker — now takes ONE ARGUMENT: subdf
# ------------------------------------------------------------
def _alignment_single_frame(subdf, radius):
    """
    Compute local velocity alignment for a single time frame.
    Returns:
        out      : np.ndarray (len(subdf),)
        idxs_out : original row indices
    """
    pts  = subdf[["x",  "y",  "z"] ].to_numpy(float)
    vels = subdf[["vx", "vy", "vz"]].to_numpy(float)

    n = len(subdf)
    out = np.full(n, np.nan, dtype=float)

    # norms
    vnorm = np.linalg.norm(vels, axis=1)
    vnorm[vnorm == 0] = np.nan

    # tree
    tree = BallTree(pts)
    nn_list = tree.query_radius(pts, r=radius, return_distance=False)

    for i_local, neigh in enumerate(nn_list):

        # only self?
        if len(neigh) <= 1:
            continue

        # drop self
        neigh = neigh[neigh != i_local]
        if len(neigh) == 0:
            continue

        vi = vels[i_local]
        ni = vnorm[i_local]
        if not np.isfinite(ni):
            continue

        vj = vels[neigh]
        nj = vnorm[neigh]

        valid = np.isfinite(nj)
        if not valid.any():
            continue

        vj = vj[valid]
        nj = nj[valid]

        dots = np.dot(vj, vi) / (ni * nj)
        out[i_local] = np.nanmean(dots)

    return out, subdf.index.to_numpy()



# ------------------------------------------------------------
# Main wrapper — same architecture as density script
# ------------------------------------------------------------
def compute_local_velocity_alignment(
        df,
        n_workers=1,
        radius=50.0,
):
    """
    Compute local velocity-alignment scores per timepoint.
    Parallelization scheme identical to compute_surface_density.
    """

    alignment = np.full(len(df), np.nan, dtype=float)

    # build per-frame dataframes (clean & simple)
    frames = [subdf for _, subdf in df.groupby("t")]

    # bind parameters to worker
    run_align = partial(_alignment_single_frame, radius=radius)

    # ------------------------------------------------------------
    # Serial path
    # ------------------------------------------------------------
    if n_workers == 1:
        for subdf in tqdm(frames, desc="velocity-align"):
            out_frame, idxs_out = run_align(subdf)
            alignment[idxs_out] = out_frame

    # ------------------------------------------------------------
    # Parallel path
    # ------------------------------------------------------------
    else:
        results = process_map(
            partial(_unwrap_helper, run_align),
            frames,
            max_workers=n_workers,
            chunksize=1,
            desc="velocity-align-par",
        )

        for out_frame, idxs_out in results:
            alignment[idxs_out] = out_frame

    return alignment



# ------------------------------------------------------------
# Worker: windowed SPEED correlation
# ------------------------------------------------------------
def _speed_corr_single_frame(subdf, full_df, window, radius):
    """
    Computes local speed–speed correlation for a single frame:

        C_i = mean_j corr(speed_i(t:t+W), speed_j(t:t+W))

    Returns:
        out      : array of correlation values (len(subdf),)
        idxs_out : global df indices (same ordering as subdf)
    """
    pts   = subdf[["x", "y", "z"]].to_numpy(float)
    tids  = subdf["track_id"].to_numpy()
    tvals = subdf["t"].to_numpy()
    n     = len(subdf)

    # ---- Build speed time-series per cell (variable length) ----
    speed_list = []
    for tid, t in zip(tids, tvals):
        mask = (full_df["track_id"] == tid) & \
               (full_df["t"].between(t - window, t + window))
        v = full_df.loc[mask, ["vx", "vy", "vz"]].to_numpy(float)
        speed_list.append(np.linalg.norm(v, axis=1))

    # ---- Pad to common length ----
    maxL = max(len(s) for s in speed_list)
    speed_ts = np.full((n, maxL), np.nan)
    for i, s in enumerate(speed_list):
        speed_ts[i, :len(s)] = s

    # ---- Neighborhood search ----
    tree = BallTree(pts)
    nn_list = tree.query_radius(pts, r=radius, return_distance=False)

    out = np.full(n, np.nan)
    for i_local, neigh in enumerate(nn_list):
        # drop self
        neigh = neigh[neigh != i_local]
        if len(neigh) == 0:
            continue

        si = speed_ts[i_local]
        valid_i = np.isfinite(si)

        cors = []
        for j in neigh:
            sj = speed_ts[j]
            mask = valid_i & np.isfinite(sj)
            if mask.sum() >= 2:
                cors.append(np.corrcoef(si[mask], sj[mask])[0, 1])

        out[i_local] = np.nanmean(cors) if len(cors) > 0 else np.nan

    return out, subdf.index.to_numpy()

def compute_local_speed_correlation(
        df,
        window=3,
        n_workers=1,
        radius=50.0,
):
    out = np.full(len(df), np.nan)
    frames = [subdf for _, subdf in df.groupby("t")]

    run = partial(_speed_corr_single_frame,
                  full_df=df,
                  window=window,
                  radius=radius)

    if n_workers == 1:
        for subdf in tqdm(frames, desc="speed-corr"):
            out_frame, idxs = run(subdf)
            out[idxs] = out_frame
    else:
        results = process_map(
            run,                    # <--- NO UNWRAP NEEDED
            frames,
            max_workers=n_workers,
            chunksize=1,
            desc="speed-corr-par",
        )
        for out_frame, idxs in results:
            out[idxs] = out_frame

    return out


# ------------------------------------------------------------
# Worker: windowed VELOCITY–vector correlation
# ------------------------------------------------------------
def _vel_corr_single_frame(subdf, full_df, window, radius):
    """
    Computes windowed velocity-vector correlation:

        C_i = mean_j corr(vec_i(t:t+W), vec_j(t:t+W))

    Velocity windows are flattened so correlation is scalar.
    """
    pts   = subdf[["x", "y", "z"]].to_numpy(float)
    tids  = subdf["track_id"].to_numpy()
    tvals = subdf["t"].to_numpy()
    n     = len(subdf)

    # ---- collect velocity windows (variable length) ----
    vel_list = []
    for tid, t in zip(tids, tvals):
        mask = (full_df["track_id"] == tid) & \
               (full_df["t"].between(t - window, t + window))
        v = full_df.loc[mask, ["vx", "vy", "vz"]].to_numpy(float)
        vel_list.append(v.reshape(-1))   # flatten

    # ---- pad ----
    maxL = max(len(v) for v in vel_list)
    vel_ts = np.full((n, maxL), np.nan)
    for i, v in enumerate(vel_list):
        vel_ts[i, :len(v)] = v

    # ---- neighbors ----
    tree = BallTree(pts)
    nn_list = tree.query_radius(pts, r=radius, return_distance=False)

    out = np.full(n, np.nan)
    for i_local, neigh in enumerate(nn_list):
        neigh = neigh[neigh != i_local]
        if len(neigh) == 0:
            continue

        vi = vel_ts[i_local]
        valid_i = np.isfinite(vi)

        cors = []
        for j in neigh:
            vj = vel_ts[j]
            mask = valid_i & np.isfinite(vj)
            if mask.sum() >= 3:   # at least one xyz triple
                cors.append(np.corrcoef(vi[mask], vj[mask])[0, 1])

        out[i_local] = np.nanmean(cors) if len(cors) > 0 else np.nan

    return out, subdf.index.to_numpy()


def compute_windowed_velocity_corr(
                                    df,
                                    window=11,
                                    n_workers=1,
                                    radius=50.0,
                                ):
    out = np.full(len(df), np.nan)
    frames = [subdf for _, subdf in df.groupby("t")]

    time_index = np.unique(df["t"].to_numpy())

    run = partial(_vel_corr_single_frame,
                  full_df=df,
                  window=window,
                  radius=radius)

    sub_window = window // 2
    if n_workers == 1:
        for t in tqdm(time_index, desc="vel-corr"):
            t_indices = (time_index <= t + sub_window) & (time_index >= t - sub_window)
            subdf = pd.concat([frames[t] for t in t_indices], ignore_index=True)
            out_frame, idxs = run(subdf)
            out[idxs] = out_frame
    else:
        results = process_map(
            run,
            frames,
            max_workers=n_workers,
            chunksize=1,
            desc="vel-corr-par",
        )
        for out_frame, idxs in results:
            out[idxs] = out_frame

    return out

