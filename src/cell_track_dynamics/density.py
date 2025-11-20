import numpy as np
from sklearn.neighbors import BallTree
from functools import partial
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


# ------------------------------------------------------------
# Needed on Windows: top-level unwrapping function
# ------------------------------------------------------------
def _unwrap_helper(func, subdf):
    return func(subdf)


# ------------------------------------------------------------
# Worker
# ------------------------------------------------------------
def _density_single_frame(
    subdf,
    sphere_df,
    radius,
    project_to_sphere,
    var_list,
):
    t = subdf["t"].iloc[0]
    sphere_row = sphere_df.loc[sphere_df["t"] == t].iloc[0]

    # positions
    pts = subdf[["x", "y", "z"]].to_numpy(float)

    # optional projection
    if project_to_sphere:
        cx = sphere_row["center_x_smooth"]
        cy = sphere_row["center_y_smooth"]
        cz = sphere_row["center_z_smooth"]  # NOTE: FIXED BUG
        R  = sphere_row["radius_smooth"]

        v = pts - np.array([cx, cy, cz])
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        unit = v / norm
        pts = unit * R

        cap_area = 2 * np.pi * R * radius
    else:
        cap_area = np.pi * radius**2

    # BallTree
    tree = BallTree(pts)

    # density
    counts = tree.query_radius(pts, r=radius, count_only=True)
    dens = counts / cap_area

    # imputation
    imputed_dict = {}
    if var_list:
        nn_list = tree.query_radius(pts, r=radius, return_distance=False)

        for col in var_list:
            vals = subdf[col].to_numpy(float)
            out = np.empty_like(vals)

            for j, neigh in enumerate(nn_list):
                nn_vals = vals[neigh]
                nn_vals = nn_vals[np.isfinite(nn_vals)]
                out[j] = nn_vals.mean() if len(nn_vals) else np.nan

            imputed_dict[col] = out

    # return with index for global writeback
    return dens, imputed_dict, subdf.index.to_numpy()


# ------------------------------------------------------------
# Main wrapper
# ------------------------------------------------------------
def compute_surface_density(
    df,
    sphere_df,
    n_workers=1,
    radius=50.0,
    project_to_sphere=False,
    var_list=None,
):
    # prepare outputs
    var_list = list(var_list) if var_list else []
    density  = np.full(len(df), np.nan)
    imputed  = {v: np.full(len(df), np.nan) for v in var_list}

    # per-frame subdataframes
    frames = [subdf for _, subdf in df.groupby("t")]

    # helper with bound args
    run_density_calc = partial(
        _density_single_frame,
        sphere_df=sphere_df,
        radius=radius,
        project_to_sphere=project_to_sphere,
        var_list=var_list,
    )

    # ------------------------------------------------------------
    # Serial
    # ------------------------------------------------------------
    if n_workers == 1:
        for subdf in tqdm(frames, desc="Calculating density statistics..."):
            dens, impt, idxs = run_density_calc(subdf)
            density[idxs] = dens
            for col, arr in impt.items():
                imputed[col][idxs] = arr

    # ------------------------------------------------------------
    # Parallel
    # ------------------------------------------------------------
    else:
        results = process_map(
            run_density_calc,
            frames,
            max_workers=n_workers,
            chunksize=1,
            desc="Calculating density statistics (parallel)...",
        )

        for dens, impt, idxs in results:
            density[idxs] = dens
            for col, arr in impt.items():
                imputed[col][idxs] = arr

    return density, imputed
