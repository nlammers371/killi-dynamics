import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
from sklearn.linear_model import LinearRegression
from src.data_io.track_io import _load_track_data, _load_tracks
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from functools import partial
from tqdm import tqdm

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

def compute_windowed_msd_fast(
    positions,
    W=40,
    step=5,
    tau_max=None,       # if None → auto rule
    n_min=20,           # minimum #pairs for τ_max
    fit_tau_min=2,
    fit_tau_max=8
):
    """
    Vectorized MSD + alpha estimation per sliding window.

    positions:  (N, 3) array for a single track

    Returns: list of dicts, one per window:
        {
            'start_idx': i,
            'end_idx': i+W,
            'taus': array([...]),
            'msd': array([...]),
            'n_obs': array([...]),
            'alpha': float,
            'r2': float,
        }
    """

    N = len(positions)
    results = []

    # ---------------------------------------------------------
    # Auto τ-max based on window size and observation cutoff
    # ---------------------------------------------------------
    if tau_max is None:
        # Want W - tau >= n_min  → tau <= W - n_min
        auto_tau = max(1, W - n_min)
        # And also a practical upper bound (¼ of the window)
        tau_max = int(min(auto_tau, W // 3))
        if tau_max < 1:
            tau_max = 1

    taus = np.arange(1, tau_max + 1)

    # ---------------------------------------------------------
    # Slide windows
    # ---------------------------------------------------------
    for i in range(0, N - W + 1, step):

        seg = positions[i:i+W]    # (W,3)

        # -----------------------------------------------------
        # FAST full pairwise squared displacement matrix
        # -----------------------------------------------------
        diffs = seg[None, :, :] - seg[:, None, :]      # (W,W,3)
        sq = np.sum(diffs * diffs, axis=2)             # (W,W)

        # -----------------------------------------------------
        # Vectorized MSD across all taus
        # -----------------------------------------------------
        msd_vals = np.zeros_like(taus, dtype=float)
        n_obs = np.zeros_like(taus, dtype=int)

        for j, tau in enumerate(taus):
            # valid pairs: sq[t, t+tau] where t = 0..W-tau-1
            block = sq[:-tau, tau:]
            if block.size == 0:
                msd_vals[j] = np.nan
                n_obs[j] = 0
            else:
                msd_vals[j] = np.mean(block)
                n_obs[j] = block.shape[0]

        # -----------------------------------------------------
        # Fit α and R² in log–log space
        # -----------------------------------------------------
        fit_mask = (
            (taus >= fit_tau_min) &
            (taus <= fit_tau_max) &
            np.isfinite(msd_vals) &
            (msd_vals > 0)
        )

        if np.sum(fit_mask) >= 3:
            X = np.log(taus[fit_mask]).reshape(-1, 1)
            y = np.log(msd_vals[fit_mask])
            lr = LinearRegression().fit(X, y)
            alpha = lr.coef_[0]
            r2 = lr.score(X, y)
        else:
            alpha = np.nan
            r2 = np.nan

        # -----------------------------------------------------
        # Store window result
        # -----------------------------------------------------
        results.append({
            'start_idx': i,
            'end_idx': i + W,
            'taus': taus.copy(),
            'msd': msd_vals,
            'n_obs': n_obs,
            'alpha': alpha,
            'r2': r2,
        })

    return results

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
        # displacements = seg[None, :, :] - seg[:, None, :]
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

def _msd_single_track(track_df, stride, window_length):
    """
    track_df: DataFrame containing a *single* track
    """
    tid = track_df["track_id"].iloc[0]
    t_vec = track_df["t"].to_numpy()
    pos = track_df[["x", "y", "z"]].to_numpy(float)

    msd_res = compute_windowed_msd(pos, W=window_length, step=stride)
    if len(msd_res) == 0:
        # no windows → return empty
        return tid, np.full_like(t_vec, np.nan, dtype=float), None

    msd_df = pd.DataFrame(msd_res)
    msd_df["track_id"] = tid

    # ---- window mid-times ----
    start_idx = msd_df["start_idx"].to_numpy()
    end_idx = msd_df["end_idx"].to_numpy()
    end_incl = np.minimum(end_idx - 1, len(track_df) - 1)
    mid_times = (t_vec[start_idx] + t_vec[end_incl]) / 2
    msd_df["time"] = mid_times

    # ---- interpolate alpha back to track time ----
    alpha_interp = np.interp(
        t_vec,
        mid_times,
        msd_df["alpha"].to_numpy(),
    )

    return tid, alpha_interp, msd_df


def compute_track_msd(
    root: Path | None,
    project_name: str | None,
    config_name: str | None,
    tracks: pd.DataFrame,
    use_flows: bool = True,
    stride: int = 5,
    window_length: int = 30,
    n_workers: int = 1
):

    # use load track data helper to load tracks
    if tracks is None:
        # _, tracking_dir = _load_tracks(root=root,
        #                                project_name=project_name,
        #                                tracking_config=config_name,
        #                                prefer_smoothed=False,
        #                                prefer_flow=use_flows)

        tracks, sphere_df = _load_track_data(
            root=root,
            project_name=project_name,
            tracking_config=config_name,
            prefer_smoothed=False,
            prefer_flow=use_flows,
        )

    tracks = tracks.sort_values(["track_id", "t"]).copy()

    # Pre-split into per-track DataFrames for efficient parallelization
    track_groups = dict(tuple(tracks.groupby("track_id")))
    valid = {tid: df for tid, df in track_groups.items() if len(df) >= window_length}

    worker = partial(_msd_single_track,
                     stride=stride,
                     window_length=window_length)

    if n_workers > 1:
        results = process_map(worker,
                              valid.values(),
                              max_workers=n_workers,
                              chunksize=1,
                              desc="Computing MSD")
    else:
        results = []
        for df in tqdm(valid.values(), desc="Computing MSD"):
            results.append(worker(df))

    # Collect results
    msd_df_list = []
    msd_df_long = tracks.loc[:, ["track_id", "t"]].copy()
    for tid, alpha_interp, msd_df in results:
        # write alpha_interp back into main DataFrame
        msd_df_long.loc[tracks["track_id"] == tid, "msd_alpha"] = alpha_interp
        if msd_df is not None:
            msd_df_list.append(msd_df)

    msd_df_full = pd.concat(msd_df_list, ignore_index=True) if msd_df_list else None

    return msd_df_long, msd_df_full




