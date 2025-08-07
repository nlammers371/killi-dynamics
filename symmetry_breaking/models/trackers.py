from pde.trackers.base import TrackerBase
import numpy as np

class NodalROITracker(TrackerBase):
    def __init__(self, grid, roi_width=500, interval=10):
        """
        Tracks Nodal distribution metrics over time in a 1D simulation.

        Parameters
        ----------
        grid : pde.CartesianGrid
            The 1D spatial grid used in the simulation.
        roi_width : float
            Width (in μm) of the region of interest centered in the domain.
        interval : int
            Number of steps between metric evaluations.
        """
        super().__init__()
        self.interval = interval
        self.grid = grid
        self.roi_width = roi_width

        self.x = grid.axes_coords[0]
        self.dx = self.x[1] - self.x[0]
        self.N_total_init = None
        self.N_roi_init = None

        x_center = 0.5 * (self.x[0] + self.x[-1])
        self.roi_mask = np.abs(self.x - x_center) <= (roi_width / 2)

        self.fraction_in_roi = []
        self.fold_change_in_roi = []
        self.fold_change_in_max = []
        self.times = []
        self.counter = 0

    def handle(self, state, time):
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
            max_change = N_max / self.N_max_init if self.N_roi_init > 0 else 0.0

            self.fraction_in_roi.append(frac_in_roi)
            self.fold_change_in_roi.append(fold_change)
            self.fold_change_in_max.append(max_change)
            self.times.append(time)

        self.counter += 1

    def get_metrics(self):
        if not self.fraction_in_roi:
            return {
                "roi_frac": None,
                "roi_fold_change": None,
                "max_fold_change": None
            }

        return {
            "roi_frac": self.fraction_in_roi[-1],
            "roi_fold_change": self.fold_change_in_roi[-1],
            "max_fold_change": self.fold_change_in_max[-1]
        }
