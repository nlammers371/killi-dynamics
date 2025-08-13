import numpy as np
from pde.trackers.base import TrackerBase

class NodalROITracker(TrackerBase):
    def __init__(self, grid, roi_width=500, interval=10,
                 save_profiles=True, store_every=None,  # None => only final; int => every Nth handle
                 downsample=None, dtype=np.float32,
                 field_labels=("Activator", "Repressor", "rho")):
        super().__init__()

        self.interval = int(interval)
        self.grid = grid
        self.roi_width = float(roi_width)

        self.save_profiles = bool(save_profiles)
        self.store_every = None if store_every in (None, 0) else int(store_every)
        self.downsample = int(downsample) if downsample else None
        self.dtype = dtype
        self.field_labels = field_labels

        self.x_full = grid.axes_coords[0]
        self.dx = float(self.x_full[1] - self.x_full[0])
        self.x = self.x_full[::self.downsample] if self.downsample else self.x_full

        # ROI mask
        x_center = 0.5 * (self.x_full[0] + self.x_full[-1])
        roi_mask_full = np.abs(self.x_full - x_center) <= (self.roi_width / 2)
        self.roi_mask = roi_mask_full[::self.downsample] if self.downsample else roi_mask_full

        # time-series metrics
        self.fraction_in_roi = []
        self.fold_change_in_roi = []
        self.fold_change_in_max = []
        self.times = []
        self.counter = 0

        # baselines
        self.N_total_init = None
        self.N_roi_init = None
        self.N_max_init = None

        # raw profile storage
        self.profiles = {}            # final profiles (numpy arrays)
        self.profile_history = {}     # only used if store_every is set

        self._last_state = None

        self.profile_history = {name: [] for name in self.field_labels}

    def _extract_profiles(self, state):
        """Return dict of raw numpy arrays (downsampled, dtype applied)."""
        out = {}
        for i, field in enumerate(state):
            name = (self.field_labels[i] if (self.field_labels and i < len(self.field_labels))
                    else getattr(field, "label", f"field_{i}"))
            arr = np.array(field.data)
            if self.downsample and self.downsample > 1:
                arr = arr[::self.downsample]
            # ensure numeric array with desired dtype
            arr = np.asarray(arr, dtype=self.dtype)
            out[name] = arr
        return out

    def handle(self, state, time):
        self._last_state = state

        # --- metrics every self.interval steps ---
        if (self.counter % self.interval) == 0:
            A = state[0].data   # Activator/Nodal total
            if self.downsample and self.downsample > 1:
                A = A[::self.downsample]
            N_total = A.sum() * (self.dx if not self.downsample else self.dx * self.downsample)
            N_max = float(np.max(A))
            N_roi = float(A[self.roi_mask].sum() * (self.dx if not self.downsample else self.dx * self.downsample))

            if self.N_total_init is None:
                self.N_total_init = N_total
                self.N_roi_init = N_roi
                self.N_max_init = N_max

            frac_in_roi = (N_roi / N_total) if N_total > 0 else 0.0
            fold_roi = (N_roi / self.N_roi_init) if self.N_roi_init > 0 else 0.0
            fold_max = (N_max / self.N_max_init) if self.N_max_init > 0 else 0.0

            self.fraction_in_roi.append(float(frac_in_roi))
            self.fold_change_in_roi.append(float(fold_roi))
            self.fold_change_in_max.append(float(fold_max))
            self.times.append(float(time))

        if self.save_profiles and (self.counter % self.interval) == 0:
            snap = self._extract_profiles(state)
            for k, v in snap.items():
                self.profile_history[k].append(v)

        # --- optional snapshot storage every k steps (raw numpy) ---
        if self.save_profiles and self.store_every and (self.counter % self.store_every) == 0:
            snap = self._extract_profiles(state)
            for k, v in snap.items():
                self.profile_history.setdefault(k, []).append(v)

        self.counter += 1

    # py-pde 0.35 passes finalize(info=...), so accept **kwargs
    def finalize(self, **kwargs):
        if self.save_profiles:
            # state = self._last_state
            # if state is not None:
            #     self.profiles = self._extract_profiles(state)
            # also expose the x-grid used for profiles
            # self.profiles["x"] = np.asarray(self.x, dtype=self.dtype)

            self.profiles = {k: np.stack(v, axis=0) for k, v in self.profile_history.items()}
            self.profiles["x"] = self.x
            self.profiles["times"] = np.asarray(self.times, dtype=self.dtype)

    def get_metrics(self):
        """Return lightweight scalars ONLY (no arrays -> pandas-safe)."""
        return {
            "roi_frac": (self.fraction_in_roi[-1] if self.fraction_in_roi else None),
            "roi_fold_change": (self.fold_change_in_roi[-1] if self.fold_change_in_roi else None),
            "max_fold_change": (self.fold_change_in_max[-1] if self.fold_change_in_max else None),
        }

    # helpers you can call after solve()
    def get_profiles(self):
        """Final profiles: dict of numpy arrays (e.g., {'Activator':..., 'Repressor':..., 'rho':..., 'x':...})."""
        return self.profiles

    def save_npz(self, path):
        """Persist final profiles (and optional history) to a compressed .npz file."""
        payload = {}
        # final
        for k, v in self.profiles.items():
            payload[f"final/{k}"] = v
        # history (optional)
        for k, lst in self.profile_history.items():
            if lst:
                payload[f"history/{k}"] = np.stack(lst, axis=0)  # (T_snapshots, X)
        np.savez_compressed(path, **payload)