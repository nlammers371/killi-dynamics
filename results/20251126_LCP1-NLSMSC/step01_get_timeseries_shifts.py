import os
from src.build_killi.run00_get_frame_shifts import get_timeseries_shifts

if __name__ == '__main__':
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project = "20251126_LCP1-NLSMSC"
    interval = 1

    get_timeseries_shifts(root=root, project_name=project, interval=interval)

