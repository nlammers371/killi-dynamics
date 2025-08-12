import numpy as np
from pde.trackers.base import TrackerBase

class NodalROITracker(TrackerBase):
    def __init__(self, grid, roi_width=500, interval=10,
                 save_profiles=True, downsample=None, dtype=np.float32,
                 field_labels=("Activator", "Repressor", "rho")):
        """
        Tracks Nodal distribution metrics over time in a 1D simulation.
        Optionally stores final spatial profiles for selected fields.

        Parameters
        ----------
        grid : pde.CartesianGrid
            The 1D spatial grid used in the simulation.
        roi_width : float
            Width (in μm) of the region of interest centered in the domain.
        interval : int
            Number of steps between metric evaluations.
        save_profiles : bool
            Whether to store final field profiles.
        downsample : int or None
            Downsampling factor for profiles (e.g. 5 keeps every 5th point).
        dtype : np.dtype
            Data type to store profiles (e.g., np.float16, np.float32).
        field_labels : tuple
            Expected labels of the fields (used to name profiles).
        """
        super().__init__()
        self.interval = interval
        self.grid = grid
        self.roi_width = roi_width

        self.save_profiles = save_profiles
        self.downsample = downsample
        self.dtype = dtype
        self.field_labels = field_labels

        self.x = grid.axes_coords[0]
        self.dx = self.x[1] - self.x[0]
        self.N_total_init = None
        self.N_roi_init = None
        self.N_max_init = None

        x_center = 0.5 * (self.x[0] + self.x[-1])
        self.roi_mask = np.abs(self.x - x_center) <= (roi_width / 2)

        self.fraction_in_roi = []
        self.fold_change_in_roi = []
        self.fold_change_in_max = []
        self.times = []
        self.counter = 0

        # Will store profiles at finalize()
        self.profiles = {}

    def handle(self, state, time):

        self._last_state = state

        if self.counter % self.interval == 0:
            A = state[0].data
            N_total = A.sum() * self.dx
            N_max = A.max()
            N_roi = A[self.roi_mask].sum() * self.dx

            if self.N_total_init is None:
                self.N_total_init = N_total
                self.N_roi_init = N_roi
                self.N_max_init = N_max

            frac_in_roi = N_roi / N_total if N_total > 0 else 0.0
            fold_change = N_roi / self.N_roi_init if self.N_roi_init > 0 else 0.0
            max_change = N_max / self.N_max_init if self.N_max_init > 0 else 0.0

            self.fraction_in_roi.append(frac_in_roi)
            self.fold_change_in_roi.append(fold_change)
            self.fold_change_in_max.append(max_change)
            self.times.append(time)

        self.counter += 1

        if not self.save_profiles:
            return
        # use provided state if any; otherwise fall back to cached last state
        st = state if state is not None else getattr(self, "_last_state", None)
        if st is None:
            return  # nothing to save (shouldn't happen in normal runs)

        for i, field in enumerate(st):
            name = (self.field_labels[i] if (self.field_labels and i < len(self.field_labels))
                    else getattr(field, "label", f"field_{i}"))
            arr = np.array(field.data)
            if self.downsample and self.downsample > 1:
                arr = arr[::self.downsample]
            arr = arr.astype(self.dtype, copy=False)
            self.profiles[f"profile_{name}"] = arr

    def get_metrics(self):
        metrics = {
            "roi_frac": self.fraction_in_roi[-1] if self.fraction_in_roi else None,
            "roi_fold_change": self.fold_change_in_roi[-1] if self.fold_change_in_roi else None,
            "max_fold_change": self.fold_change_in_max[-1] if self.fold_change_in_max else None
        }
        if self.save_profiles:
            metrics.update(self.profiles)
        return metrics
