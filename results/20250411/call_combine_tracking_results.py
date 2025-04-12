from src.build_killi.track_processing import concatenate_tracking_results
import os

if __name__ == "__main__":
    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "20250311_LCP1-NLSMSC"
    tracking_config = "tracking_20250328_redux"  # only used for ML segmentation
    # call combine tracking results
    handoff_index = 2130  # index where we hand off to the next tracking
    track_range1 = [0, 2200]
    track_range2 = [2000, 2339]  # range for the second tracking (after handoff)

    prefix = f"E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\tracking\\{project_name}\\{tracking_config}\\well0000\\"
    track_folder1 = prefix + f"track_{track_range1[0]:04}_{track_range1[1]:04}\\"
    track_folder2 = prefix + f"track_{track_range2[0]:04}_{track_range2[1]:04}\\"
    out_folder = prefix + f"track_{track_range1[0]:04}_{track_range2[1]:04}\\"

    print("Combining tracking results for project:", project_name)
    concatenate_tracking_results(track_folder1=track_folder1, track_folder2=track_folder2, out_folder=out_folder,
                                 track_range1=track_range1, track_range2=track_range2,
                                 handoff_index=handoff_index, par_flag=True, overwrite_flag=True)

    print("Done combining tracking results.")