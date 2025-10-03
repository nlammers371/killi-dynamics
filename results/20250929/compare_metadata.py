import nd2
import numpy as np
from pathlib import Path

def summarize_nd2(path):
    with nd2.ND2File(str(path)) as f:
        md = f.metadata
        sizes = dict(f.sizes or {})

        # Channel exposures
        chans = md.channels or []
        exposures = [getattr(ch, "exposureTime", None) for ch in chans]

        # Camera info
        cam = getattr(md, "camera", None)
        roiH = getattr(cam.roi, "h", None) if cam and getattr(cam, "roi", None) else None
        roiW = getattr(cam.roi, "w", None) if cam and getattr(cam, "roi", None) else None
        bit_depth = getattr(cam, "bitDepth", None)

        # Timestamps
        times = []
        for i in range(len(f)):   # safer than computing n_frames manually
            fm = f.frame_metadata(i)
            t = getattr(fm, "relativeTimeMs", None)
            if t is not None:
                times.append(t/1000.0)

        per_z_dt = None
        if "Z" in sizes and len(times) > sizes["Z"]:
            times = np.array(times)
            diffs = np.diff(times)
            per_z_dt = float(np.nanmedian(diffs))

        print(f"\nFile: {path.name}")
        print(f"  Sizes: {sizes}")
        print(f"  Exposure times (ms): {exposures}")
        print(f"  ROI (W x H): {roiW} x {roiH}")
        print(f"  Bit depth: {bit_depth}")
        print(f"  Estimated dt per Z step (s): {per_z_dt}")

# ---- edit these paths ----
exp1_dir = Path("/media/nick/cluster/projects/data/killi_tracker/raw_data/20250621/")
exp2_dir = Path("/media/nick/cluster/projects/data/killi_tracker/raw_data/20250716/")


for folder in [exp1_dir, exp2_dir]:
    print(f"\n=== {folder} ===")
    for nd2file in folder.glob("*.nd2"):
        summarize_nd2(nd2file)

nd2file1 = exp1_dir.glob("*.nd2").__next__()
file1 = nd2.ND2File(str(nd2file1))
nd2file2 = exp2_dir.glob("*.nd2").__next__()
file2 = nd2.ND2File(str(nd2file2))



# extract frame times
imObject = file1
n_z_slices = 87
n_frames_total = imObject.frame_metadata(0).contents.frameCount
frame_time_vec = [imObject.frame_metadata(i).channels[0].time.relativeTimeMs / 1000 for i in
                  range(0, n_frames_total, n_z_slices)]

# check for common nd2 artifact where time stamps jump midway through
dt_frame_approx = (imObject.frame_metadata(n_z_slices).channels[0].time.relativeTimeMs -
                   imObject.frame_metadata(0).channels[0].time.relativeTimeMs) / 1000

jump_ind = np.where(np.diff(frame_time_vec) > 2 * dt_frame_approx)[0]  # typically it is multiple orders of magnitude to large
if len(jump_ind) > 0:
    jump_ind = jump_ind[0]
    # prior to this point we will just use the time stamps. We will extrapolate to get subsequent time points
    nf = jump_ind - 1 - int(jump_ind / 2)
    dt_frame_est = (frame_time_vec[jump_ind - 1] - frame_time_vec[int(jump_ind / 2)]) / nf
    base_time = frame_time_vec[jump_ind - 1]
    for f in range(jump_ind, len(frame_time_vec)):
        frame_time_vec[f] = base_time + dt_frame_est * (f - jump_ind)
frame_time_vec = np.asarray(frame_time_vec)

print("first 10 timestamps (s):", times[:10])
print("median Δt between frames (s):", np.median(np.diff(times)))