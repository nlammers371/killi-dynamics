import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
# ---------- (unchanged) projection & graph helpers ----------

# def _spherical_overlap(uA, uB, nside=128):
#     import healpy as hp
#     pixA = hp.ang2pix(nside, np.arccos(uA[:,2]), np.arctan2(uA[:,1], uA[:,0]))
#     pixB = hp.ang2pix(nside, np.arccos(uB[:,2]), np.arctan2(uB[:,1], uB[:,0]))
#     setA, setB = set(pixA.tolist()), set(pixB.tolist())
#     inter = len(setA & setB)
#     if inter == 0: return 0.0
#     return inter / min(len(setA), len(setB))


def _spherical_overlap_iomin(uA, uB, tol_rad):
    """
    Overlap = min( |A∩B|/|A| , |A∩B|/|B| ), approximated via angular proximity.
    uA,uB: (N,3)/(M,3) unit vectors; tol_rad: angular tolerance [radians]
    """
    NA, NB = len(uA), len(uB)
    if NA == 0 or NB == 0:
        return 0.0

    chord = 2.0 * np.sin(tol_rad / 2.0)

    treeA, treeB = cKDTree(uA), cKDTree(uB)

    # A points close to any B
    hitA = np.fromiter((len(x) > 0 for x in treeB.query_ball_point(uA, r=chord)),
                       dtype=bool, count=NA)
    # B points close to any A
    hitB = np.fromiter((len(x) > 0 for x in treeA.query_ball_point(uB, r=chord)),
                       dtype=bool, count=NB)

    covA = hitA.sum() / NA  # fraction of A covered by B
    covB = hitB.sum() / NB  # fraction of B covered by A
    return float(min(covA, covB))



def _project_to_sphere(pts, center, r):
    v = pts - center
    n = np.linalg.norm(v, axis=1)
    n = np.where(n == 0, 1e-12, n)
    u = v / n[:, None]
    proj = center + r * u
    return u, proj

def _query_neighbors_on_sphere(u, angle_thresh_rad):
    chord = 2.0 * np.sin(0.5 * angle_thresh_rad)
    tree = cKDTree(u)
    return tree.query_ball_point(u, r=chord)  # lists include self

def _connected_components_from_neighbors(neigh, min_size=1):
    n = len(neigh)
    parent = np.arange(n)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    for i, nbrs in enumerate(neigh):
        for j in nbrs:
            if i != j: union(i, j)
    comps = {}
    for i in range(n):
        r = find(i)
        comps.setdefault(r, []).append(i)
    comps = [nodes for nodes in comps.values() if len(nodes) >= min_size]
    labels = np.full(n, -1, int)
    for cid, nodes in enumerate(comps):
        labels[nodes] = cid
    return labels, comps

# ---------- per-timepoint stats (adds centroid_u, fluo_mean, deg_mean, nn_dist_mean) ----------

def _cluster_stats_for_time(t_val, df_t, comps, u, r, d_thresh,
                            deg, mean_nn_dist, fluo_vals):

    recs = []
    ids = df_t.index.to_numpy()
    for cid, nodes in enumerate(comps):
        row_idx = ids[nodes]
        size = len(nodes)
        # centroid direction (unit vector on sphere)
        m = u[nodes].mean(axis=0)
        m = m / (np.linalg.norm(m) + 1e-12)

        # angular spread & spherical-cap proxy area
        ang = np.arccos(np.clip(u[nodes] @ m, -1.0, 1.0))
        theta_mean = float(ang.mean())
        theta_max  = float(ang.max()) if size > 1 else 0.0
        area_cap   = 2.0 * np.pi * (r**2) * (1.0 - np.cos(theta_max))

        # aggregate metrics
        fluo_mean    = float(np.nanmean(fluo_vals[nodes])) if size else np.nan
        deg_mean     = float(np.mean(deg[nodes])) if size else 0.0
        nn_dist_mean = float(np.nanmean(mean_nn_dist[nodes])) if size else np.nan

        recs.append({
            "r": r,
            "d_thresh": d_thresh,
            "t": t_val,
            "cluster_id_local": cid,
            "size": size,
            "theta_mean": theta_mean,
            "theta_max": theta_max,
            "area_cap": area_cap,
            "fluo_mean": fluo_mean,
            "deg_mean": deg_mean,
            "nn_dist_mean": nn_dist_mean,
            "member_index": row_idx,
            "member_positions": u,
            # "member_track_id": df_t.loc[row_idx, "track_id"].to_numpy(),
            "centroid_u": m,   # NEW: unit-vector centroid for gating
        })
    return recs

def find_clusters_per_timepoint(
    tracks_df: pd.DataFrame,
    sphere_df: pd.DataFrame,
    d_thresh: float = 30,
    min_size: int = 5,
    time_col: str = "t",
    xcol: str = "x", ycol: str = "y", zcol: str = "z",
    fluo_col: str = "mean_fluo",
    sphere_time_col: str = "t",
    sphere_center_cols=("center_x_smoothed", "center_x_smoothed", "center_x_smoothed"),
    sphere_radius_col="r_smoothed"
) -> Dict[Any, List[dict]]:
    sph = sphere_df.set_index(sphere_time_col)[list(sphere_center_cols)+[sphere_radius_col]]
    clusters_by_t: Dict[Any, List[dict]] = {}

    for t_val, df_t in tqdm(tracks_df.groupby(time_col, sort=True), desc="Finding clusters per timepoint"):
        if t_val not in sph.index:
            continue
        xc, yc, zc = sph.loc[t_val, list(sphere_center_cols)].values
        r = float(sph.loc[t_val, sphere_radius_col])

        pts = df_t[[xcol, ycol, zcol]].to_numpy(float)
        u, _proj = _project_to_sphere(pts, np.array([xc, yc, zc], float), r)

        # ε-graph
        theta = d_thresh / r
        neigh = _query_neighbors_on_sphere(u, theta)

        # node degree & mean neighbor distance
        n = len(neigh)
        deg = np.fromiter((max(0, len(N)-1) for N in neigh), dtype=int, count=n)
        mean_nn_dist = np.full(n, np.nan, float)
        for i, N in enumerate(neigh):
            Nn = [j for j in N if j != i]
            if not Nn: continue
            dots = u[i] @ u[Nn].T
            dots = np.clip(dots, -1.0, 1.0)
            theta_ij = np.arccos(dots)
            mean_nn_dist[i] = float(np.mean(r * theta_ij))

        labels, comps = _connected_components_from_neighbors(neigh, min_size=min_size)
        fluo_vals = df_t[fluo_col].to_numpy(dtype=float) if fluo_col in df_t.columns else np.full(len(df_t), np.nan)

        recs = _cluster_stats_for_time(t_val, df_t, comps, u, r, d_thresh,
                                       deg, mean_nn_dist, fluo_vals)
        clusters_by_t[t_val] = recs

    return clusters_by_t

# ---------- new linker with metric choice, centroid gate, and merge handling ----------

def _centroid_angle(u1, u2):
    d = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    return np.arccos(d)


def _geodesic_extrapolate(u_prevprev, u_prev, step=1.0):
    if u_prevprev is None or u_prev is None:
        return u_prev
    axis = np.cross(u_prevprev, u_prev)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return u_prev
    axis /= n
    ang = _centroid_angle(u_prevprev, u_prev)
    theta = step * ang
    k = axis;
    v = u_prev
    v_rot = (v * np.cos(theta) +
             np.cross(k, v) * np.sin(theta) +
             k * np.dot(k, v) * (1 - np.cos(theta)))
    v_rot /= (np.linalg.norm(v_rot) + 1e-12)
    return v_rot


def _overlap_coeff(a_ids, b_ids):
    sa, sb = set(a_ids.tolist()), set(b_ids.tolist())
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / min(len(sa), len(sb))


def _jaccard(a_ids, b_ids):
    sa, sb = set(a_ids.tolist()), set(b_ids.tolist())
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    uni = len(sa | sb)
    return inter / uni if uni > 0 else 0.0


def _feat_vec(c, w_size=1.0, w_fluo=1.0, w_deg=1.0, w_nnd=1.0):
    return np.array([
        w_size * np.log1p(max(c.get("size", np.nan), 0)),
        w_fluo * c.get("fluo_mean", np.nan),
        w_deg * c.get("deg_mean", np.nan),
        w_nnd * c.get("nn_dist_mean", np.nan)
    ], dtype=float)

def _set_sim(prev_members, curr_members, metric: str):
    sa, sb = set(prev_members.tolist()), set(curr_members.tolist())
    if not sa and not sb: return 1.0
    inter = len(sa & sb)
    if inter == 0: return 0.0
    if metric == "overlap":
        return inter / min(len(sa), len(sb))
    elif metric == "jaccard":
        return inter / len(sa | sb)
    else:
        raise ValueError("metric must be 'overlap' or 'jaccard'")


def _feat_cos(a_row, b_row, **wkwargs):
    av = _feat_vec(a_row, **wkwargs)
    bv = _feat_vec(b_row, **wkwargs)
    m = np.isfinite(av) & np.isfinite(bv)
    if not np.any(m):
        return 0.0
    av, bv = av[m], bv[m]
    na, nb = np.linalg.norm(av), np.linalg.norm(bv)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def track_clusters_over_time(
    clusters_by_t: Dict[Any, List[dict]],
    link_metric: str = "overlap",   # "overlap" or "jaccard"
    sim_min: float = 0.3,           # min set‑similarity to consider a link
    max_centroid_angle: Optional[float] = None,  # radians; None disables hard gate
    w_sim: float = 1.0,             # weight for set similarity
    w_feat: float = 0.5,            # weight for feature cosine
    w_pred: float = 0.5,            # weight for prediction proximity term
    pred_step: float = 1.0,         # extrapolation step (frames)
    carry_merge_parents: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Motion‑ and feature‑aware linker with merge handling.
    Returns (cluster_ts, merges_df).
    """
    times = sorted(clusters_by_t.keys())
    next_pid = 0
    active: Dict[int, dict] = {}               # pid -> last record
    prev_centroid: Dict[int, Optional[np.ndarray]] = {}
    prevprev_centroid: Dict[int, Optional[np.ndarray]] = {}
    rows = []
    merges = []

    for i, t in enumerate(tqdm(times, desc="Linking clusters over time")):
        curr = clusters_by_t[t]

        if i == 0:
            for c in curr:
                pid = next_pid; next_pid += 1
                active[pid] = c
                prev_centroid[pid] = c["centroid_u"]
                prevprev_centroid[pid] = None
                rows.append({**c, "cluster_id": pid, "merged_from": []})
            continue

        prev_pids = list(active.keys())
        P, C = len(prev_pids), len(curr)

        if P == 0:
            for c in curr:
                pid = next_pid; next_pid += 1
                active[pid] = c
                prev_centroid[pid] = c["centroid_u"]
                prevprev_centroid[pid] = None
                rows.append({**c, "cluster_id": pid, "merged_from": []})
            continue

        # Build combined score matrix
        # Score = w_sim * set_sim + w_feat * feat_cos + w_pred * pred_prox
        S = np.full((P, C), -np.inf, float)
        for a, pid in enumerate(prev_pids):
            pa = active[pid]
            u_pred = _geodesic_extrapolate(prevprev_centroid[pid], prev_centroid[pid], step=pred_step)
            for b, cb in enumerate(curr):
                if max_centroid_angle is not None:
                    ang = _centroid_angle(pa["centroid_u"], cb["centroid_u"])
                    if ang > max_centroid_angle:
                        continue
                # if "member_track_id" not in pa or "member_track_id" not in cb:
                theta = cb["d_thresh"] / (0.5*cb["r"] + 0.5*pa["r"])
                set_sim = _spherical_overlap_iomin(pa["member_positions"], cb["member_positions"], tol_rad=theta)
                # else:
                #     set_sim = _set_sim(pa["member_track_id"], cb["member_track_id"], link_metric)
                if set_sim < sim_min:
                    continue
                feat = _feat_cos(pa, cb)
                # prediction proximity as cosine of angle between prediction and candidate centroid
                pred_cos = float(np.clip(np.dot(u_pred, cb["centroid_u"]), -1.0, 1.0)) if u_pred is not None else 0.0
                score = w_sim*set_sim + w_feat*feat + w_pred*pred_cos
                S[a, b] = score

        # Greedy assign by descending score (more stable than Hungarian with mixed terms)
        pairs = []
        used_prev, used_curr = set(), set()
        # flatten valid scores
        flat = [(S[a, b], a, b) for a in range(P) for b in range(C) if np.isfinite(S[a, b])]
        flat.sort(reverse=True)
        for sc, a, b in flat:
            if a in used_prev or b in used_curr:
                continue
            pairs.append((a, b, sc))
            used_prev.add(a); used_curr.add(b)

        # Link primaries; detect merges (other qualifying prev that point to same curr)
        # Build reverse map: for each curr b, find all prev a with acceptable similarity
        supporters = {b: [] for b in range(C)}
        for a in range(P):
            for b in range(C):
                if np.isfinite(S[a, b]):
                    supporters[b].append(a)

        # Apply links
        for a, b, _ in pairs:
            pid = prev_pids[a]
            c = curr[b]
            # continue track
            active[pid] = c
            rows.append({**c, "cluster_id": pid, "merged_from": []})
            # update motion buffers
            prevprev_centroid[pid] = prev_centroid[pid]
            prev_centroid[pid] = c["centroid_u"]

            # MERGES: any other supporters (not the primary) get merged/closed
            others = [aa for aa in supporters[b] if aa != a]
            if others and carry_merge_parents:
                parents = []
                for aa in others:
                    pid_other = prev_pids[aa]
                    # close track if still active and not linked elsewhere
                    if pid_other in active:
                        merges.append({"t": t, "merged_into": pid, "merged_from": pid_other})
                        active.pop(pid_other, None)
                        parents.append(pid_other)
                if parents:
                    rows[-1]["merged_from"] = parents

        # Unassigned current → new tracks
        for b, c in enumerate(curr):
            if b not in used_curr:
                pid = next_pid; next_pid += 1
                active[pid] = c
                prev_centroid[pid] = c["centroid_u"]
                prevprev_centroid[pid] = None
                rows.append({**c, "cluster_id": pid, "merged_from": []})

        # Any prev not updated this frame are ended
        for a, pid in enumerate(prev_pids):
            if a not in used_prev and pid in active and active[pid]["t"] < t:
                active.pop(pid, None)
                prev_centroid.pop(pid, None)
                prevprev_centroid.pop(pid, None)

    # Assemble TS + summaries
    df_ts = pd.DataFrame(rows)
    stats = (df_ts.groupby("cluster_id")
             .agg(start_t=("t","min"), end_t=("t","max"),
                  duration=("t", lambda v: v.max()-v.min()+1),
                  mean_size=("size","mean"),
                  max_size=("size","max"),
                  mean_fluo=("fluo_mean","mean"),
                  mean_deg=("deg_mean","mean"),
                  mean_nn_dist=("nn_dist_mean","mean"),
                  n_obs=("t","count"))
             .reset_index())
    df_ts = df_ts.merge(stats, on="cluster_id", how="left")
    merges_df = pd.DataFrame(merges, columns=["t","merged_into","merged_from"])
    return df_ts, merges_df


# ---------- main stitcher ----------

def stitch_tracklets(
        cluster_ts: pd.DataFrame,
        gap_max: int = 2,  # max temporal gap (frames) to bridge
        window: int = 0,  # +/- frames around endpoints to search for best overlap
        link_metric: str = "overlap",  # "overlap" or "jaccard"
        sim_min: float = 0.3,  # min set-similarity to consider a stitch
        max_centroid_angle: float | None = np.deg2rad(20),  # None disables gate
        w_sim: float = 1.0,  # weight: set similarity
        w_feat: float = 0.5,  # weight: feature cosine
        w_pred: float = 0.5,  # weight: prediction proximity
        pred_step: float = 1.0,  # step for geodesic extrapolation
        w_size: float = 2.0,  # size up-weight inside feature vector
        max_iters: int = 3  # repeat stitching passes
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stitch fragmented cluster tracklets across small temporal gaps by remapping child
    tracklet cluster_ids to parent ones using a multi-term score.

    Returns
    -------
    stitched_ts : cluster_ts with 'cluster_id_stitched' column
    stitch_log  : DataFrame with [parent_id, child_id, parent_seg, child_seg, score, delta_t]
    """

    df = cluster_ts.copy()

    # Build per-cluster contiguous segments (tracklets)
    seg_rows = []
    for cid, d in tqdm(df.groupby("cluster_id"), desc="Stitching tracklets"):
        tt = d.sort_values("t")
        times = tt["t"].to_numpy()
        # segment breaks where t jumps > 1
        breaks = np.where(np.diff(times) > 1)[0]
        # indices for segment boundaries
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(times) - 1]
        for s_idx, e_idx in zip(starts, ends):
            seg = tt.iloc[s_idx:e_idx + 1].copy()
            seg_rows.append({
                "cluster_id": cid,
                "seg_idx": len([r for r in seg_rows if r["cluster_id"] == cid]),  # local index
                "t_start": int(seg["t"].iloc[0]),
                "t_end": int(seg["t"].iloc[-1]),
                "len": int(seg.shape[0]),
                "rows": seg  # keep the slice for boundary access
            })
    seg_df = pd.DataFrame(seg_rows)

    # Current mapping from original cluster_id to stitched id (initially identity)
    id_map = {cid: cid for cid in df["cluster_id"].unique()}

    # Utility: get set similarity at boundaries with optional window
    def boundary_set_sim(a_rows: pd.DataFrame, b_rows: pd.DataFrame) -> float:
        # choose metric
        simfun = _overlap_coeff if link_metric == "overlap" else _jaccard
        a_times = a_rows["t"].to_numpy()
        b_times = b_rows["t"].to_numpy()
        # target endpoints
        a_end = a_times.max()
        b_start = b_times.min()
        best = 0.0
        for dt_a in range(-window, window + 1):
            for dt_b in range(-window, window + 1):
                ta = a_end + dt_a
                tb = b_start + dt_b
                if ta in a_times and tb in b_times:
                    A = a_rows.loc[a_rows["t"] == ta, "member_track_id"].iloc[0]
                    B = b_rows.loc[b_rows["t"] == tb, "member_track_id"].iloc[0]
                    best = max(best, simfun(A, B))
        return best

    # Utility: feature cosine between end of A and start of B
    def boundary_feat_cos(a_rows: pd.DataFrame, b_rows: pd.DataFrame) -> float:
        ar = a_rows.loc[a_rows["t"] == a_rows["t"].max()].iloc[0].to_dict()
        br = b_rows.loc[b_rows["t"] == b_rows["t"].min()].iloc[0].to_dict()
        return _feat_cos(ar, br, w_size=w_size)

    # Utility: prediction proximity using last two centroids of A and first of B
    def boundary_pred_cos(a_rows: pd.DataFrame, b_rows: pd.DataFrame) -> float:
        a_sorted = a_rows.sort_values("t")
        if a_sorted.shape[0] >= 2:
            u_prevprev = np.array(a_sorted.iloc[-2]["centroid_u"])
        else:
            u_prevprev = None
        u_prev = np.array(a_sorted.iloc[-1]["centroid_u"])
        u_pred = _geodesic_extrapolate(u_prevprev, u_prev, step=pred_step)
        u_b = np.array(b_rows.sort_values("t").iloc[0]["centroid_u"])
        return float(np.clip(np.dot(u_pred, u_b), -1.0, 1.0))

    stitch_events = []

    for it in range(max_iters):
        stitched_any = False

        # Build candidate pair list (A ends before B starts; gap ≤ gap_max)
        cand = []
        for ia, A in seg_df.iterrows():
            for ib, B in seg_df.iterrows():
                if A["cluster_id"] == B["cluster_id"]:
                    continue
                if A["t_end"] < B["t_start"]:
                    gap = B["t_start"] - A["t_end"] - 1
                    if 0 <= gap <= gap_max:
                        # centroid gate (optional)
                        if max_centroid_angle is not None:
                            uA = np.array(A["rows"].iloc[-1]["centroid_u"])
                            uB = np.array(B["rows"].iloc[0]["centroid_u"])
                            if _centroid_angle(uA, uB) > max_centroid_angle:
                                continue
                        # set similarity (required min)
                        sim = boundary_set_sim(A["rows"], B["rows"])
                        if sim < sim_min:
                            continue
                        feat = boundary_feat_cos(A["rows"], B["rows"])
                        pred = boundary_pred_cos(A["rows"], B["rows"])
                        score = w_sim * sim + w_feat * feat + w_pred * pred
                        cand.append((score, ia, ib, gap, sim, feat, pred))

        if not cand:
            break

        # Greedy, highest score first, without conflicts (each seg can be used once per iter)
        cand.sort(reverse=True, key=lambda t: t[0])
        used_A, used_B = set(), set()
        for score, ia, ib, gap, sim, feat, pred in cand:
            if ia in used_A or ib in used_B:
                continue
            A = seg_df.loc[ia];
            B = seg_df.loc[ib]

            # Remap child (B.cluster_id) → parent (A.cluster_id)
            parent_id = id_map[A["cluster_id"]]
            child_id = id_map[B["cluster_id"]]
            if parent_id == child_id:
                used_A.add(ia);
                used_B.add(ib)
                continue

            # Apply relabel in df and seg_df
            df.loc[df["cluster_id"] == child_id, "cluster_id"] = parent_id
            id_map = {old: (parent_id if new == child_id else new)
                      for old, new in id_map.items()}

            # mark and log
            used_A.add(ia);
            used_B.add(ib)
            stitched_any = True
            stitch_events.append({
                "iter": it,
                "parent_id": parent_id,
                "child_id": child_id,
                "parent_seg": (int(A["cluster_id"]), int(A["seg_idx"])),
                "child_seg": (int(B["cluster_id"]), int(B["seg_idx"])),
                "score": float(score),
                "gap": int(gap),
                "sim": float(sim),
                "feat_cos": float(feat),
                "pred_cos": float(pred)
            })

        if not stitched_any:
            break

        # Recompute segments after relabeling (so next iteration can chain)
        seg_rows = []
        for cid, d in df.groupby("cluster_id"):
            tt = d.sort_values("t")
            times = tt["t"].to_numpy()
            breaks = np.where(np.diff(times) > 1)[0]
            starts = np.r_[0, breaks + 1]
            ends = np.r_[breaks, len(times) - 1]
            for s_idx, e_idx in zip(starts, ends):
                seg = tt.iloc[s_idx:e_idx + 1].copy()
                seg_rows.append({
                    "cluster_id": cid,
                    "seg_idx": len([r for r in seg_rows if r["cluster_id"] == cid]),
                    "t_start": int(seg["t"].iloc[0]),
                    "t_end": int(seg["t"].iloc[-1]),
                    "len": int(seg.shape[0]),
                    "rows": seg
                })
        seg_df = pd.DataFrame(seg_rows)

    # Recompute per-cluster summaries and attach new ID
    stitched_ts = df.copy()
    stitched_ts = (stitched_ts
                   .rename(columns={"cluster_id": "cluster_id_stitched"}))
    stats = (stitched_ts.groupby("cluster_id_stitched")
             .agg(start_t=("t", "min"), end_t=("t", "max"),
                  duration=("t", lambda v: v.max() - v.min() + 1),
                  mean_size=("size", "mean"),
                  max_size=("size", "max"),
                  mean_fluo=("fluo_mean", "mean"),
                  mean_deg=("deg_mean", "mean"),
                  mean_nn_dist=("nn_dist_mean", "mean"),
                  n_obs=("t", "count"))
             .reset_index())
    stitched_ts = stitched_ts.merge(stats, on="cluster_id_stitched", how="left")

    stitch_log = pd.DataFrame(stitch_events)
    return stitched_ts, stitch_log

