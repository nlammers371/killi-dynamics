# 🧩 Cluster Tracking and Stitching Pipeline

This module identifies clusters of cells on a spherical surface, tracks those clusters over time,  
and stitches fragmented tracklets into continuous trajectories.

---

## Overview

The pipeline operates in **three stages**:

1. **Cluster Detection** – group individual cell detections into spatial clusters at each timepoint  
2. **Cluster Tracking** – link clusters across consecutive frames based on geometry, features, and motion prediction  
3. **Tracklet Stitching** – reconnect fragmented cluster tracks across small temporal gaps

All stages share a single configuration object (`ClusterTrackingConfig`) that defines  
hyperparameters for geometry thresholds, feature weighting, and linking gates.

---

## 1️⃣ Cluster Detection (`find_clusters_per_timepoint`)

**Goal:** identify spatially coherent cell clusters on a spherical surface.

### Inputs
- `tracks_df`: per-cell data containing at least time (`t`) and 3D coordinates (`x, y, z`)
- `sphere_df`: time-indexed table of spherical reference geometry (`center_x/y/z_smoothed`, `r_smoothed`)
- `config.d_thresh`: distance threshold (µm) converted to angular radius on the sphere
- `config.min_size`: minimum number of points per cluster

### How it works
1. Project each cell’s position to the embryo’s reference sphere.
2. Build an **ε-graph** on the unit-sphere using angular proximity.
3. Compute **connected components** of that graph → clusters.
4. For each cluster:
   - Centroid direction (`centroid_u`)
   - Angular spread (`theta_mean`, `theta_max`)
   - Mean fluorescence, neighbor degree, neighbor distance
   - Optional member track IDs (if present)

The result is a dictionary `clusters_by_t[time] → list of cluster records`.

---

## 2️⃣ Cluster Tracking (`track_clusters_over_time`)

**Goal:** link clusters between consecutive frames into temporal trajectories.

### Key ideas
- **Geometric proximity** replaces the old shared-particle method.  
  Two clusters are linked if their spherical footprints overlap within a tolerance.
- **Feature similarity** compares size, mean fluorescence, degree, and neighbor distance.
- **Motion prediction** extrapolates the previous centroid direction along a geodesic.

### Combined score
\[
\text{score} = w_\mathrm{sim} \cdot \text{Overlap} +
                w_\mathrm{feat} \cdot \text{FeatureCosine} +
                w_\mathrm{pred} \cdot \text{PredictionProximity}
\]

Greedy assignment selects highest-scoring pairs, with optional merge detection if multiple
previous clusters converge on the same new one.

### Important parameters
| Parameter | Meaning |
|------------|----------|
| `sim_min` | Minimum geometric similarity to consider a link |
| `max_centroid_angle` | Maximum angular separation (radians) allowed between centroids |
| `w_sim`, `w_feat`, `w_pred` | Weights for each score term |
| `pred_step` | Extrapolation step length (frames) |

Outputs:
- `cluster_ts`: per-frame cluster table with assigned `cluster_id`
- `merges_df`: list of merge events

---

## 3️⃣ Tracklet Stitching (`stitch_tracklets`)

**Goal:** reconnect short cluster tracklets separated by brief temporal gaps.

### Logic
1. Break each `cluster_id` into contiguous segments (no time gaps).
2. For each candidate pair (A ends before B starts):
   - Check time gap ≤ `config.gap_max`
   - Optional centroid-angle gate
   - Compute geometric overlap between endpoints (via `_spherical_overlap_iomin`)
   - Compute feature cosine and predicted motion alignment
3. Compute the same weighted score and greedily join non-conflicting pairs.
4. Repeat up to `config.max_iters` times to allow chained stitching.

### Parameters
| Parameter | Description |
|------------|--------------|
| `gap_max` | Maximum allowed gap (frames) |
| `window` | Temporal window ± around endpoints to evaluate overlap |
| `max_iters` | Number of stitching passes |
| `w_size` | Size weight used inside feature vector |
| (others) | Same as tracking stage |

Outputs:
- `stitched_ts`: full table with `cluster_id_stitched`
- `stitch_log`: list of stitch events (parent/child IDs, gap, score components)

---

## ⚙️ Configuration (`ClusterTrackingConfig`)

All tunable parameters live in this dataclass:

```python
@dataclass
class ClusterTrackingConfig:
    link_metric: str = "overlap"
    sim_min: float = 0.3
    max_centroid_angle: float | None = np.deg2rad(20)
    pred_step: float = 1.0
    gap_max: int = 2
    window: int = 0
    max_iters: int = 3
    w_sim: float = 1.0
    w_feat: float = 0.5
    w_pred: float = 0.5
    w_size: float = 2.0
    d_thresh: float = 30.0
    min_size: int = 5
