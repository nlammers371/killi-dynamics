from __future__ import annotations
import numbers
from pathlib import Path
from typing import Tuple, Literal

import numpy as np
import pandas as pd
import zarr

# -------------------- optional GPU backend --------------------
try:
    import cupy as cp
    _DEFAULT_XP = cp
    _HAS_GPU = True
    print("[VirtualFuseArray] Using CuPy backend.")
except Exception:
    _DEFAULT_XP = np
    _HAS_GPU = False
    print("[VirtualFuseArray] CuPy not found; using NumPy backend.")


class VirtualFuseArray:
    """
    Virtual fused view over a two-sided Zarr store without duplicating data.
    Preserves full C,Z,Y,X indexing semantics from the CPU version.
    """

    def __init__(
        self,
        store_path: str | Path,
        moving_side: str = "side_01",
        reference_side: str = "side_00",
        overlap_z: int = 30,
        use_gpu: bool = True,
        interp: Literal["nearest", "linear"] = "nearest",
    ):
        self.store_path = Path(store_path)
        self.root = zarr.open_group(self.store_path, mode="r")

        self.ref_name = reference_side
        self.mov_name = moving_side
        self.ref = self.root[self.ref_name]
        self.mov = self.root[self.mov_name]

        self.overlap_z = int(overlap_z)
        self.interp = interp
        self.use_gpu = bool(use_gpu and _HAS_GPU)
        self.xp = _DEFAULT_XP if self.use_gpu else np

        # Metadata
        self.ref_meta = self._parse_side_meta(self.ref.attrs)
        self.mov_meta = self._parse_side_meta(self.mov.attrs)
        self.ref_tf = self._load_transform(self.ref.attrs.get("rigid_transform", {}))
        self.mov_tf = self._load_transform(self.mov.attrs.get("rigid_transform", {}))

        # Basic checks
        if self.ref.shape[0] != self.mov.shape[0]:
            raise ValueError("Time dimension (T) differs between sides.")

        # Channels & shapes
        self.C = self._side_channels(self.ref.shape, self.ref_meta)
        C_mov = self._side_channels(self.mov.shape, self.mov_meta)
        if C_mov != self.C:
            raise ValueError(f"Channel count mismatch: {self.C} vs {C_mov}")

        Zr, Yr, Xr = self._side_spatial(self.ref.shape, self.ref_meta)
        Zm, Ym, Xm = self._side_spatial(self.mov.shape, self.mov_meta)
        self.dtype = self.ref.dtype
        self.shape = (self.ref.shape[0], self.C, Zr + Zm, Yr, Xr)
        self.attrs = dict(self.root.attrs)

    # ---------------- Metadata helpers ---------------- #
    @staticmethod
    def _parse_side_meta(attrs) -> dict:
        dim_order = str(attrs.get("dim_order", "TZYX")).upper()
        if dim_order not in ("TZYX", "TCZYX"):
            raise ValueError(f"Unsupported dim_order: {dim_order}")
        has_c = "C" in dim_order
        idx = {ax: dim_order.index(ax) for ax in dim_order}
        return {"dim_order": dim_order, "has_c": has_c, "idx": idx}

    @staticmethod
    def _side_channels(shape: Tuple[int, ...], meta: dict) -> int:
        return 1 if meta["dim_order"] == "TZYX" else shape[meta["idx"]["C"]]

    @staticmethod
    def _side_spatial(shape: Tuple[int, ...], meta: dict) -> Tuple[int, int, int]:
        return tuple(shape[meta["idx"][ax]] for ax in "ZYX")

    def _load_transform(self, tdict: dict) -> dict:
        flip = np.array(tdict.get("flip", [False, False, False]), dtype=bool)
        shifts = None
        path = tdict.get("per_frame_shifts", None)
        if isinstance(path, str):
            csv_path = self.store_path / path
            df = pd.read_csv(csv_path)
            if not {"frame", "zs", "ys", "xs"}.issubset(df.columns):
                raise ValueError(f"Shift CSV missing columns: {csv_path}")
            shifts = df[["zs", "ys", "xs"]].to_numpy(dtype=float)
        return {"flip": flip, "per_frame": shifts}

    # ---------------- Math helpers ---------------- #
    @staticmethod
    def _split_shift(vec3):
        v = np.asarray(vec3, dtype=float)
        vi = np.floor(v).astype(int)
        vf = v - vi
        return vi, vf

    def _lin_shift_along_axis(self, arr, frac, axis):
        if frac == 0.0:
            return arr
        xp = self.xp
        s0, s1 = [slice(None)] * arr.ndim, [slice(None)] * arr.ndim
        s0[axis], s1[axis] = slice(0, -1), slice(1, None)
        a0, a1 = arr[tuple(s0)], arr[tuple(s1)]
        blended = (1 - frac) * a0 + frac * a1
        tail = xp.take(blended, [-1], axis=axis)
        return xp.concatenate([blended, tail], axis=axis)

    def _apply_fractional_shift_zyx(self, arr, fz, fy, fx):
        xp = self.xp
        out = arr.astype(xp.float32, copy=False)
        if fz != 0:
            out = self._lin_shift_along_axis(out, fz, 1)
        if fy != 0:
            out = self._lin_shift_along_axis(out, fy, 2)
        if fx != 0:
            out = self._lin_shift_along_axis(out, fx, 3)
        return out

    # ---------------- Native ROI read ---------------- #
    def _read_side_roi_czyx(
        self, side_group: zarr.core.Group, meta: dict,
        t: int, c_sel, z_slice, y_slice, x_slice
    ):
        order, idx = meta["dim_order"], meta["idx"]
        if order == "TZYX":
            sl = [slice(None)] * 4
            sl[idx["T"]], sl[idx["Z"]], sl[idx["Y"]], sl[idx["X"]] = t, z_slice, y_slice, x_slice
            arr = np.asarray(side_group[tuple(sl)])
            arr = arr[np.newaxis, ...]
        else:
            sl = [slice(None)] * 5
            sl[idx["T"]], sl[idx["C"]] = t, c_sel
            sl[idx["Z"]], sl[idx["Y"]], sl[idx["X"]] = z_slice, y_slice, x_slice
            arr = np.asarray(side_group[tuple(sl)])
            if arr.ndim == 3:
                arr = arr[np.newaxis, ...]
        return self.xp.asarray(arr) if self.use_gpu else arr

    # ---------------- Fusion core ---------------- #
    def _fuse_time_roi(self, t, c_sel, zf_slice, yf_slice, xf_slice):
        xp = self.xp
        Zr, Yr, Xr = self._side_spatial(self.ref.shape, self.ref_meta)
        Zm, Ym, Xm = self._side_spatial(self.mov.shape, self.mov_meta)
        full_Z = Zr + Zm
        z0, z1, _ = zf_slice.indices(full_Z)
        y0, y1, _ = yf_slice.indices(Yr)
        x0, x1, _ = xf_slice.indices(Xr)
        Z_len, Y_len, X_len = z1 - z0, y1 - y0, x1 - x0

        shift = np.zeros(3, float)
        if self.mov_tf["per_frame"] is not None and t < len(self.mov_tf["per_frame"]):
            shift[:] = self.mov_tf["per_frame"][t]
        (dzi, dyi, dxi), (dzf, dyf, dxf) = self._split_shift(shift)
        fz, fy, fx = self.mov_tf["flip"].tolist()

        # z positioning
        ref_z0, ref_z1 = Zm, Zm + Zr
        mov_start = Zm - dzi
        mov_z0, mov_z1 = mov_start, mov_start + Zm
        rf0, rf1 = max(z0, ref_z0), min(z1, ref_z1)
        mf0, mf1 = max(z0, mov_z0), min(z1, mov_z1)
        rz0, rz1 = max(0, rf0 - Zm), max(0, rf1 - Zm)
        mz0, mz1 = max(0, mf0 - mov_start), max(0, mf1 - mov_start)
        r_out0, r_out1 = max(0, rf0 - z0), max(0, rf1 - z0)
        m_out0, m_out1 = max(0, mf0 - z0), max(0, mf1 - z0)

        # Y/X
        my0, my1 = max(0, y0 - dyi), max(0, y0 - dyi + (y1 - y0))
        mx0, mx1 = max(0, x0 - dxi), max(0, x0 - dxi + (x1 - x0))
        m_out_y0, m_out_y1 = max(0, dyi), max(0, dyi + (y1 - y0))
        m_out_x0, m_out_x1 = max(0, dxi), max(0, dxi + (x1 - x0))

        ref_buf = xp.zeros((self.C, Z_len, Y_len, X_len), xp.float32)
        mov_buf = xp.zeros_like(ref_buf)

        if rz1 > rz0:
            ref_block = self._read_side_roi_czyx(self.ref, self.ref_meta, t, c_sel,
                                                 slice(rz0, rz1), slice(y0, y1), slice(x0, x1))
            ref_buf[:, r_out0:r_out1] = ref_block.astype(xp.float32, copy=False)
        if mz1 > mz0:
            mov_block = self._read_side_roi_czyx(self.mov, self.mov_meta, t, c_sel,
                                                 slice(mz0, mz1), slice(my0, my1), slice(mx0, mx1))
            mov_block = mov_block.astype(xp.float32, copy=False)
            sl = (slice(None),
                  slice(None, None, -1) if fz else slice(None),
                  slice(None, None, -1) if fy else slice(None),
                  slice(None, None, -1) if fx else slice(None))
            mov_block = mov_block[sl]
            if self.interp == "linear":
                mov_block = self._apply_fractional_shift_zyx(mov_block, dzf, dyf, dxf)
            mov_buf[:, m_out0:m_out1, m_out_y0:m_out_y1, m_out_x0:m_out_x1] = mov_block

        fused = ref_buf + mov_buf
        ov0, ov1 = max(r_out0, m_out0), min(r_out1, m_out1)
        if ov1 > ov0:
            L = ov1 - ov0
            ramp = min(L, max(1, self.overlap_z))
            s = ov0 + (L - ramp) // 2
            e = s + ramp
            w = xp.linspace(0, 1, ramp, dtype=xp.float32)[None, :, None, None]
            a, b = ref_buf[:, s:e], mov_buf[:, s:e]
            fused[:, s:e] = (1 - w) * a + w * b
        return fused.astype(self.dtype, copy=False)

    # ---------------- Indexing ---------------- #
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = list(key) + [slice(None)] * (5 - len(key))

        t_sel = key[0]
        if isinstance(t_sel, numbers.Integral):
            t_idx, squeeze_t = [int(t_sel)], True
        elif isinstance(t_sel, slice):
            t_idx, squeeze_t = list(range(*t_sel.indices(self.shape[0]))), False
        else:
            t_idx, squeeze_t = list(t_sel), len(t_sel) == 1

        c_sel = key[1]
        z_sel, y_sel, x_sel = key[2], key[3], key[4]
        frames = [self._fuse_time_roi(t, c_sel, z_sel, y_sel, x_sel) for t in t_idx]
        xp = self.xp
        out = xp.stack(frames, axis=0)
        if squeeze_t:
            out = out[0]
        if out.ndim >= 4 and out.shape[-4] == 1:
            out = xp.squeeze(out, axis=-4)
        return out

    def __repr__(self):
        backend = "GPU(CuPy)" if self.use_gpu else "CPU(NumPy)"
        return (f"<VirtualFuseArray shape={self.shape} dtype={self.dtype} "
                f"sides=({self.ref_name}, {self.mov_name}) "
                f"overlap_z={self.overlap_z} interp='{self.interp}' backend={backend}>")
