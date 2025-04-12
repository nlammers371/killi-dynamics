from tqdm import tqdm
from src.build_killi.track_processing import track_fluorescence_wrapper


if __name__ == "__main__":

    # script to stitch tracks after initial tracking. Also updates corresponding seg_zarr's
    # At this point, should have tracked all relevant experiments

    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"

    project_name_list = ["20250311_LCP1-NLSMSC", "20250311_LCP1-NLSMSC"]#, "20250311_LCP1-NLSMSC"]  #, "20250311_LCP1-NLSMSC", "20250311_LCP1-NLSMSC_marker"]
    use_marker_list = [False, True] #, False]  #, False, True]
    track_range_list = [[2000, 2339], [1200, 2339]]#, [0, 2200]]  #"track_2000_2339", "track_1200_2339", "track_0000_2200"]#, "track_2000_2339", "track_1200_2339"]
    track_config_list = ["tracking_20250328_redux", "tracking_20250328_redux"]#, "tracking_20250328_redux"]  #, "tracking_20250328_redux", "tracking_20250328_redux"]
    overwrite = True
    par_flag = True
    # set gap closing parameters
    # overwrite = False

    for i in tqdm(range(len(project_name_list)), desc="Processing projects", unit="project"):

        project_name = project_name_list[i]
        tracking_config = track_config_list[i]
        tracking_range = track_range_list[i]
        marker_flag = use_marker_list[i]

        track_fluorescence_wrapper(root, project_name, tracking_config, suffix="", well_num=0, start_i=tracking_range[0],
                                   fluo_channel=None, par_flag=par_flag, stop_i=tracking_range[1],
                                   use_marker_masks=marker_flag, overwrite=True)
