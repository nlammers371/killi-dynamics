from src.tracking import combine_tracking_results
import os

if __name__ == "__main__":
    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "20250311_LCP1-NLSMSC"
    tracking_config = "tracking_20250328_redux"  # only used for ML segmentation
    # call combine tracking results
    handoff_index = 2130  # index where we hand off to the next tracking
    track_range1 = [0, 2339]
    track_range2 = [2000, 2339]  # range for the second tracking (after handoff)
    print("Combining tracking results for project:", project_name)
    combine_tracking_results(root, project_name, tracking_config, track_range1=track_range1, track_range2=track_range2,
                             handoff_index=handoff_index, par_flag=False, overwrite_flag=False)

    print("Done combining tracking results.")