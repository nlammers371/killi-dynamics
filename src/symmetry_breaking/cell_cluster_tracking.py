import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from dataclasses import dataclass
# ---------- (unchanged) projection & graph helpers ----------

# def _spherical_overlap(uA, uB, nside=128):
#     import healpy as hp
#     pixA = hp.ang2pix(nside, np.arccos(uA[:,2]), np.arctan2(uA[:,1], uA[:,0]))
#     pixB = hp.ang2pix(nside, np.arccos(uB[:,2]), np.arctan2(uB[:,1], uB[:,0]))
#     setA, setB = set(pixA.tolist()), set(pixB.tolist())
#     inter = len(setA & setB)
#     if inter == 0: return 0.0
#     return inter / min(len(setA), len(setB))

@dataclass
class ClusterTrackingConfig:
    # ---- Linking and gating ----
    link_metric: str = "overlap"     # geometric or set-based
    sim_min: float = 0.4
    max_centroid_angle: Optional[float] = np.deg2rad(20)
    pred_step: float = 1.0
    gap_max: int = 2
    window: int = 0
    max_iters: int = 3

    # ---- Weights ----
    w_sim: float = 1.0
    w_feat: float = 0.5
    w_pred: float = 0.5
    w_size: float = 2.0

    # ---- Geometry ----
    d_thresh: float = 25.0
    min_size: int = 5


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
        if "track_id" in df_t.columns:
            member_track_ids = df_t.loc[row_idx, "track_id"].to_numpy()
        else:
            member_track_ids = None

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
            "member_positions": u[nodes],
            "member_track_id": member_track_ids,
            "centroid_u": m,   # NEW: unit-vector centroid for gating
        })
    return recs

def find_clusters_per_timepoint(
    tracks_df: pd.DataFrame,
    sphere_df: pd.DataFrame,
    config: Optional[ClusterTrackingConfig] = None,
    time_col: str = "t",
    xcol: str = "x", ycol: str = "y", zcol: str = "z",
    fluo_col: str = "mean_fluo",
    sphere_time_col: str = "t",
    sphere_center_cols=("center_x_smoothed", "center_y_smoothed", "center_z_smoothed"),
    sphere_radius_col="r_smoothed"
) -> Dict[Any, List[dict]]:
    """Detect spatial clusters on the spherical surface for each timepoint."""

    if config is None:
        config = ClusterTrackingConfig()

    d_thresh = config.d_thresh
    min_size = config.min_size

    sph = sphere_df.set_index(sphere_time_col)[list(sphere_center_cols)+[sphere_radius_col]]
    clusters_by_t: Dict[Any, List[dict]] = {}

    for t_val, df_t in tqdm(tracks_df.groupby(time_col, sort=True),
                            desc="Finding clusters per timepoint"):
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
            if not Nn:
                continue
            dots = u[i] @ u[Nn].T
            dots = np.clip(dots, -1.0, 1.0)
            theta_ij = np.arccos(dots)
            mean_nn_dist[i] = float(np.mean(r * theta_ij))

        labels, comps = _connected_components_from_neighbors(neigh, min_size=min_size)
        fluo_vals = (df_t[fluo_col].to_numpy(dtype=float)
                     if fluo_col in df_t.columns else np.full(len(df_t), np.nan))

        recs = _cluster_stats_for_time(
            t_val, df_t, comps, u, r, d_thresh, deg, mean_nn_dist, fluo_vals
        )
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
    config: Optional[ClusterTrackingConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Link clusters across timepoints using geometric proximity and feature similarity."""

    if config is None:
        config = ClusterTrackingConfig()

    link_metric = config.link_metric
    sim_min = config.sim_min
    max_centroid_angle = config.max_centroid_angle
    w_sim, w_feat, w_pred = config.w_sim, config.w_feat, config.w_pred
    pred_step = config.pred_step
    carry_merge_parents = True  # retain same behavior

    times = sorted(clusters_by_t.keys())
    next_pid = 0
    active: Dict[int, dict] = {}
    prev_centroid: Dict[int, Optional[np.ndarray]] = {}
    prevprev_centroid: Dict[int, Optional[np.ndarray]] = {}
    rows = []
    merges = []

    for i, t in enumerate(tqdm(times, desc="Linking clusters over time")):
        curr = clusters_by_t[t]

        if i == 0 or not active:
            for c in curr:
                pid = next_pid
                next_pid += 1
                active[pid] = c
                prev_centroid[pid] = c["centroid_u"]
                prevprev_centroid[pid] = None
                rows.append({**c, "cluster_id": pid, "merged_from": []})
            continue

        prev_pids = list(active.keys())
        P, C = len(prev_pids), len(curr)
        S = np.full((P, C), -np.inf, float)

        for a, pid in enumerate(prev_pids):
            pa = active[pid]
            u_pred = _geodesic_extrapolate(prevprev_centroid[pid],
                                           prev_centroid[pid],
                                           step=pred_step)
            for b, cb in enumerate(curr):
                if max_centroid_angle is not None:
                    ang = _centroid_angle(pa["centroid_u"], cb["centroid_u"])
                    if ang > max_centroid_angle:
                        continue

                # geometric overlap
                theta = cb["d_thresh"] / (0.5 * cb["r"] + 0.5 * pa["r"])
                set_sim = _spherical_overlap_iomin(pa["member_positions"],
                                                   cb["member_positions"],
                                                   tol_rad=theta)
                if set_sim < sim_min:
                    continue

                feat = _feat_cos(pa, cb)
                pred_cos = float(np.clip(np.dot(u_pred, cb["centroid_u"]),
                                         -1.0, 1.0)) if u_pred is not None else 0.0
                score = w_sim * set_sim + w_feat * feat + w_pred * pred_cos
                S[a, b] = score

        # greedy assignment
        pairs = []
        used_prev, used_curr = set(), set()
        flat = [(S[a, b], a, b) for a in range(P) for b in range(C)
                if np.isfinite(S[a, b])]
        flat.sort(reverse=True)
        for sc, a, b in flat:
            if a in used_prev or b in used_curr:
                continue
            pairs.append((a, b, sc))
            used_prev.add(a)
            used_curr.add(b)

        # build reverse supporters map
        supporters = {b: [] for b in range(C)}
        for a in range(P):
            for b in range(C):
                if np.isfinite(S[a, b]):
                    supporters[b].append(a)

        # apply links
        for a, b, _ in pairs:
            pid = prev_pids[a]
            c = curr[b]
            active[pid] = c
            rows.append({**c, "cluster_id": pid, "merged_from": []})
            prevprev_centroid[pid] = prev_centroid[pid]
            prev_centroid[pid] = c["centroid_u"]

            # merges
            others = [aa for aa in supporters[b] if aa != a]
            if others and carry_merge_parents:
                parents = []
                for aa in others:
                    pid_other = prev_pids[aa]
                    if pid_other in active:
                        merges.append({"t": t, "merged_into": pid, "merged_from": pid_other})
                        active.pop(pid_other, None)
                        parents.append(pid_other)
                if parents:
                    rows[-1]["merged_from"] = parents

        # new clusters
        for b, c in enumerate(curr):
            if b not in used_curr:
                pid = next_pid
                next_pid += 1
                active[pid] = c
                prev_centroid[pid] = c["centroid_u"]
                prevprev_centroid[pid] = None
                rows.append({**c, "cluster_id": pid, "merged_from": []})

        # deactivate old
        for a, pid in enumerate(prev_pids):
            if a not in used_prev and pid in active and active[pid]["t"] < t:
                active.pop(pid, None)
                prev_centroid.pop(pid, None)
                prevprev_centroid.pop(pid, None)

    # assemble TS and summaries
    df_ts = pd.DataFrame(rows)
    stats = (df_ts.groupby("cluster_id")
             .agg(start_t=("t", "min"), end_t=("t", "max"),
                  duration=("t", lambda v: v.max()-v.min()+1),
                  mean_size=("size", "mean"),
                  max_size=("size", "max"),
                  mean_fluo=("fluo_mean", "mean"),
                  mean_deg=("deg_mean", "mean"),
                  mean_nn_dist=("nn_dist_mean", "mean"),
                  n_obs=("t", "count"))
             .reset_index())
    df_ts = df_ts.merge(stats, on="cluster_id", how="left")
    merges_df = pd.DataFrame(merges, columns=["t","merged_into","merged_from"])
    return df_ts, merges_df



# ---------- main stitcher ----------

def stitch_tracklets(
    cluster_ts: pd.DataFrame,
    config: Optional[ClusterTrackingConfig] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if config is None:
        config = ClusterTrackingConfig()

    sim_min = config.sim_min
    max_centroid_angle = config.max_centroid_angle
    gap_max = config.gap_max
    window = config.window  # currently only used inside boundary_* if you add it back
    w_sim, w_feat, w_pred, w_size = (
        config.w_sim,
        config.w_feat,
        config.w_pred,
        config.w_size,
    )
    pred_step = config.pred_step
    max_iters = config.max_iters
    d_thresh = config.d_thresh

    df = cluster_ts.copy()

    # ---------- initial segment build ----------
    seg_rows = []
    for cid, d in tqdm(df.groupby("cluster_id"), desc="Stitching tracklets"):
        tt = d.sort_values("t")
        times = tt["t"].to_numpy()
        breaks = np.where(np.diff(times) > 1)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(times) - 1]
        for s_idx, e_idx in zip(starts, ends):
            seg = tt.iloc[s_idx : e_idx + 1].copy()
            seg_rows.append(
                {
                    "cluster_id": cid,
                    "seg_idx": len([r for r in seg_rows if r["cluster_id"] == cid]),
                    "t_start": int(seg["t"].iloc[0]),
                    "t_end": int(seg["t"].iloc[-1]),
                    "len": int(seg.shape[0]),
                    "rows": seg,
                }
            )
    seg_df = pd.DataFrame(seg_rows).reset_index(drop=True)

    # map from current cluster_id → stitched id (starts as identity)
    id_map = {cid: cid for cid in df["cluster_id"].unique()}

    # ---------- helper to (re)build caches from seg_df ----------
    def _build_seg_cache(seg_df: pd.DataFrame):
        seg_array = seg_df.to_dict("records")
        t_start = np.array([s["t_start"] for s in seg_array])
        t_end = np.array([s["t_end"] for s in seg_array])
        # endpoint centroids
        u_end = [np.array(s["rows"].iloc[-1]["centroid_u"]) for s in seg_array]
        u_start = [np.array(s["rows"].iloc[0]["centroid_u"]) for s in seg_array]
        return seg_array, t_start, t_end, u_end, u_start

    seg_array, t_start, t_end, u_end, u_start = _build_seg_cache(seg_df)

    # ---------- boundary utilities ----------
    def boundary_set_sim(a_rows, b_rows) -> float:
        ta = a_rows["t"].max()
        tb = b_rows["t"].min()
        A = np.vstack(a_rows.loc[a_rows["t"] == ta, "member_positions"].iloc[0])
        B = np.vstack(b_rows.loc[b_rows["t"] == tb, "member_positions"].iloc[0])
        r_mean = 0.5 * (a_rows["r"].iloc[0] + b_rows["r"].iloc[0])
        theta = d_thresh / r_mean
        return _spherical_overlap_iomin(A, B, tol_rad=theta)

    def boundary_feat_cos(a_rows: pd.DataFrame, b_rows: pd.DataFrame) -> float:
        ar = a_rows.loc[a_rows["t"] == a_rows["t"].max()].iloc[0].to_dict()
        br = b_rows.loc[b_rows["t"] == b_rows["t"].min()].iloc[0].to_dict()
        return _feat_cos(ar, br, w_size=w_size)

    def boundary_pred_cos(a_rows: pd.DataFrame, b_rows: pd.DataFrame) -> float:
        a_sorted = a_rows.sort_values("t")
        u_prevprev = (
            np.array(a_sorted.iloc[-2]["centroid_u"])
            if a_sorted.shape[0] >= 2
            else None
        )
        u_prev = np.array(a_sorted.iloc[-1]["centroid_u"])
        u_pred = _geodesic_extrapolate(u_prevprev, u_prev, step=pred_step)
        u_b = np.array(b_rows.sort_values("t").iloc[0]["centroid_u"])
        return float(np.clip(np.dot(u_pred, u_b), -1.0, 1.0))

    stitch_events = []

    # ---------- main stitching loop ----------
    for it in range(max_iters):
        stitched_any = False
        cand = []

        order = np.argsort(t_end)
        for idx_a in tqdm(
            order, desc=f"Stitch pass {it+1}/{max_iters}", leave=False
        ):
            tA_end = t_end[idx_a]

            # only look at segments that start shortly after this one ends
            mask = (t_start > tA_end) & (t_start <= tA_end + gap_max + 1)
            if not np.any(mask):
                continue

            for idx_b in np.where(mask)[0]:
                # optional centroid gate
                if max_centroid_angle is not None:
                    ang = _centroid_angle(u_end[idx_a], u_start[idx_b])
                    if ang > max_centroid_angle:
                        continue

                A_rows = seg_array[idx_a]["rows"]
                B_rows = seg_array[idx_b]["rows"]

                sim = boundary_set_sim(A_rows, B_rows)
                if sim < sim_min:
                    continue

                feat = boundary_feat_cos(A_rows, B_rows)
                pred = boundary_pred_cos(A_rows, B_rows)
                score = w_sim * sim + w_feat * feat + w_pred * pred
                gap = t_start[idx_b] - tA_end - 1
                cand.append((score, idx_a, idx_b, gap, sim, feat, pred))

        if not cand:
            break

        # pick non-conflicting best candidates
        cand.sort(reverse=True, key=lambda t: t[0])
        used_A, used_B = set(), set()

        for score, ia, ib, gap, sim, feat, pred in cand:
            if ia in used_A or ib in used_B:
                continue

            # NOTE: use iloc for both, since ia/ib are positional
            A = seg_df.iloc[ia]
            B = seg_df.iloc[ib]

            parent_id = id_map[A["cluster_id"]]
            child_id = id_map[B["cluster_id"]]
            if parent_id == child_id:
                used_A.add(ia)
                used_B.add(ib)
                continue

            # relabel in main df
            df.loc[df["cluster_id"] == child_id, "cluster_id"] = parent_id
            # update id_map so future references to child's id map to parent
            id_map = {
                old: (parent_id if new == child_id else new)
                for old, new in id_map.items()
            }

            used_A.add(ia)
            used_B.add(ib)
            stitched_any = True
            stitch_events.append(
                {
                    "iter": it,
                    "parent_id": parent_id,
                    "child_id": child_id,
                    "parent_seg": (int(A["cluster_id"]), int(A["seg_idx"])),
                    "child_seg": (int(B["cluster_id"]), int(B["seg_idx"])),
                    "score": float(score),
                    "gap": int(gap),
                    "sim": float(sim),
                    "feat_cos": float(feat),
                    "pred_cos": float(pred),
                }
            )

        if not stitched_any:
            break

        # ---------- rebuild segments AFTER relabeling ----------
        seg_rows = []
        for cid, d in df.groupby("cluster_id"):
            tt = d.sort_values("t")
            times = tt["t"].to_numpy()
            breaks = np.where(np.diff(times) > 1)[0]
            starts = np.r_[0, breaks + 1]
            ends = np.r_[breaks, len(times) - 1]
            for s_idx, e_idx in zip(starts, ends):
                seg = tt.iloc[s_idx : e_idx + 1].copy()
                seg_rows.append(
                    {
                        "cluster_id": cid,
                        "seg_idx": len(
                            [r for r in seg_rows if r["cluster_id"] == cid]
                        ),
                        "t_start": int(seg["t"].iloc[0]),
                        "t_end": int(seg["t"].iloc[-1]),
                        "len": int(seg.shape[0]),
                        "rows": seg,
                    }
                )
        seg_df = pd.DataFrame(seg_rows).reset_index(drop=True)

        seg_array, t_start, t_end, u_end, u_start = _build_seg_cache(seg_df)

    # ---------- final formatting ----------
    stitched_ts = df.rename(columns={"cluster_id": "cluster_id_stitched"}).copy()
    stats = (
        stitched_ts.groupby("cluster_id_stitched")
        .agg(
            start_t=("t", "min"),
            end_t=("t", "max"),
            duration=("t", lambda v: v.max() - v.min() + 1),
            mean_size=("size", "mean"),
            max_size=("size", "max"),
            mean_fluo=("fluo_mean", "mean"),
            mean_deg=("deg_mean", "mean"),
            mean_nn_dist=("nn_dist_mean", "mean"),
            n_obs=("t", "count"),
        )
        .reset_index()
    )
    stitched_ts = stitched_ts.merge(stats, on="cluster_id_stitched", how="left")
    stitch_log = pd.DataFrame(stitch_events)
    return stitched_ts, stitch_log



