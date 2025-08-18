import numpy as np
from pde.trackers.base import TrackerBase

class NodalROITracker(TrackerBase):
    def __init__(self, grid, roi_width=500, interval=10,
                 save_profiles=True, store_every=None,  # None => only final; int => every Nth handle
                 downsample=None, dtype=np.float32,
                 field_labels=("Activator", "Repressor", "rho"),
                 blip_logger=None):  # <--- NEW: shared event list
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

        # pulse logging
        self._blip_logger = blip_logger if blip_logger is not None else []
        self._blip_cursor = 0  # keeps track of how many events we've already read

        self.pulses = {
            "Activator": {"time": [], "x": [], "size": [], "sigma": [], "mode": []},
            "Repressor": {"time": [], "x": [], "size": [], "sigma": [], "mode": []},
        }

        # raw profile storage
        self.profiles = {}            # final profiles (numpy arrays)
        self.profile_history = {}     # only used if store_every is set

        self._last_state = None
        self.profile_history = {name: [] for name in self.field_labels}

        # --- NEW: pulse logging support ---
        # Expect a list of dicts appended by the PDE like:
        # {"time": t, "channel": "Activator"|"Repressor", "x": x0, "amp": A, "sigma": s, "mode":"impulse"|"direct"}
        self._blip_logger = blip_logger if blip_logger is not None else []
        self._blip_cursor = 0  # index of the next unseen event

        # store pulses by channel
        self.pulses = {
            "Activator": {"time": [], "x": [], "size": [], "sigma": [], "mode": []},
            "Repressor": {"time": [], "x": [], "size": [], "sigma": [], "mode": []},
        }

    def _extract_profiles(self, state):
        """Return dict of raw numpy arrays (downsampled, dtype applied)."""
        out = {}
        for i, field in enumerate(state):
            name = (self.field_labels[i] if (self.field_labels and i < len(self.field_labels))
                    else getattr(field, "label", f"field_{i}"))
            arr = np.array(field.data)
            if self.downsample and self.downsample > 1:
                arr = arr[::self.downsample]
            arr = np.asarray(arr, dtype=self.dtype)  # ensure numeric array with desired dtype
            out[name] = arr
        return out

    def _ingest_new_pulses(self):
        """Read new entries from the shared blip logger and store them."""
        new_entries = self._blip_logger[self._blip_cursor:]
        if not new_entries:
            return

        for ev in new_entries:
            ch = "Activator" if str(ev.get("channel", "")).lower().startswith(("a", "n")) else "Repressor"
            self.pulses[ch]["time"].append(float(ev.get("time", np.nan)))
            self.pulses[ch]["x"].append(float(ev.get("x", np.nan)))
            self.pulses[ch]["size"].append(float(ev.get("amp", ev.get("size", np.nan))))
            self.pulses[ch]["sigma"].append(float(ev.get("sigma", np.nan)) if ev.get("sigma") is not None else np.nan)
            self.pulses[ch]["mode"].append(ev.get("mode", ""))

        self._blip_cursor += len(new_entries)

    def handle(self, state, time):
        self._last_state = state

        # --- NEW: first, ingest any new pulses that occurred up to this handle ---
        self._ingest_new_pulses()

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

    def finalize(self, **kwargs):
        # Ingest any pulses appended after the last handle() call
        self._ingest_new_pulses()

        if self.save_profiles:
            self.profiles = {k: np.stack(v, axis=0) for k, v in self.profile_history.items()}
            self.profiles["x"] = self.x
            self.profiles["times"] = np.asarray(self.times, dtype=self.dtype)

        # Convert pulse lists to numpy arrays for convenience
        for ch in ("Activator", "Repressor"):
            for key in ("time", "x", "size", "sigma"):
                self.pulses[ch][key] = np.asarray(self.pulses[ch][key], dtype=float)
            # mode stays as list of strings

    def get_metrics(self):
        """Return lightweight scalars ONLY (no arrays -> pandas-safe)."""
        return {
            "roi_frac": (self.fraction_in_roi[-1] if self.fraction_in_roi else None),
            "roi_fold_change": (self.fold_change_in_roi[-1] if self.fold_change_in_roi else None),
            "max_fold_change": (self.fold_change_in_max[-1] if self.fold_change_in_max else None),
        }

    def get_profiles(self):
        """Final profiles and time grid used for them."""
        return self.profiles

    # --- NEW: expose pulses for post-hoc analysis ---
    def get_pulses(self, channel=None, as_dict=True):
        """
        Return logged pulses.
        - channel: None => both; "Activator" or "Repressor" for one channel.
        - as_dict=True => nested dict; False => (times, x, size, sigma, mode) tuple(s).
        """
        if channel is None:
            return self.pulses if as_dict else (
                (self.pulses["Activator"]["time"], self.pulses["Activator"]["x"],
                 self.pulses["Activator"]["size"], self.pulses["Activator"]["sigma"],
                 self.pulses["Activator"]["mode"]),
                (self.pulses["Repressor"]["time"], self.pulses["Repressor"]["x"],
                 self.pulses["Repressor"]["size"], self.pulses["Repressor"]["sigma"],
                 self.pulses["Repressor"]["mode"])
            )
        ch = "Activator" if channel.lower().startswith(("a", "n")) else "Repressor"
        return self.pulses[ch] if as_dict else (
            self.pulses[ch]["time"], self.pulses[ch]["x"],
            self.pulses[ch]["size"], self.pulses[ch]["sigma"],
            self.pulses[ch]["mode"]
        )

    def save_npz(self, path):
        """Persist final profiles, history, and pulses to a compressed .npz file."""
        payload = {}
        # final profiles
        for k, v in self.profiles.items():
            payload[f"final/{k}"] = v
        # history (optional)
        for k, lst in self.profile_history.items():
            if lst:
                payload[f"history/{k}"] = np.stack(lst, axis=0)  # (T_snapshots, X)
        # pulses
        for ch in ("Activator", "Repressor"):
            payload[f"pulses/{ch}/time"] = np.asarray(self.pulses[ch]["time"])
            payload[f"pulses/{ch}/x"] = np.asarray(self.pulses[ch]["x"])
            payload[f"pulses/{ch}/size"] = np.asarray(self.pulses[ch]["size"])
            payload[f"pulses/{ch}/sigma"] = np.asarray(self.pulses[ch]["sigma"])
            # store mode as a fixed-width string array to keep .npz simple
            if len(self.pulses[ch]["mode"]) > 0:
                payload[f"pulses/{ch}/mode"] = np.array(self.pulses[ch]["mode"], dtype="U16")
            else:
                payload[f"pulses/{ch}/mode"] = np.array([], dtype="U16")

        np.savez_compressed(path, **payload)
