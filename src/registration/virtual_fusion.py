from __future__ import annotations
import numbers
from pathlib import Path
from typing import Tuple, Literal
from scipy.ndimage import gaussian_filter
import numpy as np
import pandas as pd
import zarr
import dask.array as da
from dask import delayed

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
if _HAS_GPU:
    import cupy as cp
    from cupyx.scipy import ndimage as sp_nd
    _meta_scalar = cp.array((), dtype=np.float32)
else:
    from scipy import ndimage as sp_nd
    _meta_scalar = np.array((), dtype=np.float32)

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
        eval_mode: bool = False,
        use_gpu: bool = True,
        # seam_sigma_px: tuple = (1.0, 0.5, 0.5),
        interp: Literal["nearest", "linear"] = "nearest",
    ):
        self.store_path = Path(store_path)
        self.root = zarr.open_group(self.store_path, mode="r")

        self.ref_name = reference_side
        self.mov_name = moving_side
        self.ref = self.root[self.ref_name]
        self.mov = self.root[self.mov_name]
        self.eval_mode = eval_mode
        # self.seam_sigma_px = seam_sigma_px
        self.interp = interp
        if self.eval_mode:
            self.interp = "nearest"
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

        # add other useful attributes
        self.attrs['voxel_size_um'] = self.ref.attrs.get('voxel_size_um', None)
        self.attrs['time_resolution_s'] = self.ref.attrs.get('time_resolution_s', None)
        self.attrs["dim_order"] = self.ref.attrs.get("dim_order", None)
        self.attrs["channels"] = self.ref.attrs.get("channels", None)
        self.attrs["raw_voxel_scale_um"] = self.ref.attrs.get("raw_voxel_scale_um", None)


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

    # put this in the class (near other helpers)
    def _normalize_c_sel(self, side_group, meta, c_sel):
        """Return (indexer, C_len, squeeze_c). Accepts int/slice/iterable.
           For TZYX (no channels), always returns slice(None), 1, squeeze_c=(c_sel is int).
        """
        import numbers
        if meta["dim_order"] == "TZYX":
            # single-channel data: pretend C=1 and optionally squeeze later if user gave an int
            squeeze_c = isinstance(c_sel, numbers.Integral)
            return slice(None), 1, squeeze_c

        # TCZYX
        C_total = side_group.shape[meta["idx"]["C"]]
        squeeze_c = isinstance(c_sel, numbers.Integral)

        if isinstance(c_sel, numbers.Integral):
            if not (0 <= int(c_sel) < C_total):
                raise IndexError(f"channel index {int(c_sel)} out of range 0..{C_total - 1}")
            return int(c_sel), 1, True

        if isinstance(c_sel, slice):
            start, stop, step = c_sel.indices(C_total)
            C_len = max(0, (stop - start + (abs(step) - 1)) // (step if step > 0 else -step))
            return slice(start, stop, step), C_len, False

        # iterable (list/array) of channels
        idx = list(c_sel)
        if len(idx) == 0:
            return [], 0, False
        # (optional) bounds check
        if not all(0 <= int(i) < C_total for i in idx):
            raise IndexError("channel index out of range")
        return idx, len(idx), False


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

    def _fuse_time_roi(self, t, c_sel, zf_slice, yf_slice, xf_slice):
        xp = self.xp
        Zr, Yr, Xr = self._side_spatial(self.ref.shape, self.ref_meta)
        Zm, Ym, Xm = self._side_spatial(self.mov.shape, self.mov_meta)
        full_Z = Zr + Zm

        # --- normalize channel selection ONCE and reuse for both sides ---
        c_indexer_ref, C_len, squeeze_c = self._normalize_c_sel(self.ref, self.ref_meta, c_sel)
        # we assume sides have matching channels; reuse same c_indexer for mov
        c_indexer_mov = c_indexer_ref

        # fused ROI
        z0, z1, _ = zf_slice.indices(full_Z)
        y0, y1, _ = yf_slice.indices(Yr)
        x0, x1, _ = xf_slice.indices(Xr)
        Z_len, Y_len, X_len = full_Z, Yr, Xr #z1 - z0, y1 - y0, x1 - x0

        # shifts / flips as before ...
        shift = np.zeros(3, float)
        if self.mov_tf["per_frame"] is not None and t < len(self.mov_tf["per_frame"]):
            shift[:] = self.mov_tf["per_frame"][t]
        (dzi, dyi, dxi), (dzf, dyf, dxf) = self._split_shift(shift)
        fz, fy, fx = self.mov_tf["flip"].tolist()
        # fx = True
        # fy = True
        # z placement + native/fused ranges ... (keep your current logic)
        # ref_z0, ref_z1 = Zm, Zm + Zr
        # mov_z0, mov_z1 = dzi, dzi + Zm

        # REFERENCE indexing
        rz0, rz1 = 0, Zr
        r_out_z0, r_out_z1 = Zm, Zr + Zm
        # if self.eval_mode:
        #     rz0 = rz0 - dzi
        #     r_out_z0 = r_out_z0 - dzi


        # MOVING indexing
        mz0, mz1 = max(0, -dzi), Zm
        m_out_z0, m_out_z1 = max(0, dzi), max(0, dzi) + (mz1 - mz0)
        # if self.eval_mode:
        #     m_out_z0 = m_out_z0 + dzi

        # no Y (assume non inverted)
        ys = -1
        my0, my1 = max(0, ys * dyi), min(Ym, Ym + ys * dyi)
        m_out_y0 = max(0, ys * dyi)
        m_out_y1 = max(0, ys * dyi) + (my1 - my0)

        # no X (assume inverted)
        xs = 1
        mx0, mx1 = max(0, xs * dxi), min(Xm, Xm + xs * dxi)
        m_out_x0, m_out_x1 = max(0, xs * dxi), max(0, xs * dxi) + (mx1 - mx0)

        # small helper to create a lazily-read block
        def _delayed_read_ref():
            # returns xp array, float32, shape: (C_len, rz1-rz0, Yr, Xr)
            arr = self._read_side_roi_czyx(
                self.ref, self.ref_meta, t, c_indexer_ref,
                slice(rz0, rz1), slice(0, Yr), slice(0, Xr)
            ).astype(xp.float32, copy=False)
            return arr

        def _delayed_read_mov():
            # returns xp array, float32, shape: (C_len, mz1-mz0, my1-my0, mx1-mx0)
            arr = self._read_side_roi_czyx(
                self.mov, self.mov_meta, t, c_indexer_mov,
                slice(mz0, mz1), slice(my0, my1), slice(mx0, mx1)
            ).astype(xp.float32, copy=False)
            return arr

        dtype = xp.float32
        # lazily make reference block (or empty)
        if rz1 > rz0:
            # 1️⃣ use xp dtype for delayed blocks
            ref_block = da.from_delayed(
                delayed(_delayed_read_ref)(),
                shape=(C_len, rz1 - rz0, Yr, Xr),
                dtype=dtype,
                meta=_meta_scalar
            )
        else:
            ref_block = da.zeros((C_len, 0, Yr, Xr), dtype=np.float32)

        # lazily make moving block (or empty)
        if mz1 > mz0:
            mov_block = da.from_delayed(
                delayed(_delayed_read_mov)(),
                shape=(C_len, mz1 - mz0, my1 - my0, mx1 - mx0),
                dtype=dtype,
                meta=_meta_scalar
            )
        else:
            mov_block = da.zeros((C_len, 0, 0, 0), dtype=np.float32)

        # flips (lazy slicing)
        sl = (
            slice(None),
            slice(None, None, -1) if fz else slice(None),
            slice(None, None, -1) if fy else slice(None),
            slice(None, None, -1) if fx else slice(None),
        )
        mov_block = mov_block[sl]

        # fractional shift (only if linear)
        def _shift_block(arr, dzf, dyf, dxf):
            # 'arr' will be a numpy or cupy array depending on backend
            # order=1, mode="nearest" to match your logic
            return sp_nd.shift(arr, shift=(0, dzf, dyf, dxf), order=1, mode="nearest")

        if self.interp == "linear" and mov_block.shape[1] > 0:
            mov_block = da.map_blocks(
                _shift_block, mov_block,
                dtype=np.float32,
                dzf=dzf, dyf=dyf, dxf=dxf,
                meta=_meta_scalar
            )

        # ---- place blocks into the full fused canvas lazily via padding ----
        # reference placement: pad Z to [r_out_z0:r_out_z1], Y to [0:Yr], X to [0:Xr]
        # pad spec: ((C_pre,C_post),(Z_pre,Z_post),(Y_pre,Y_post),(X_pre,X_post))
        def _pad_to_canvas(block, z0, z1, y0, y1, x0, x1):
            block = block.map_blocks(xp.asarray)  # enforce backend
            z_pre, z_post = z0, Z_len - z1
            y_pre, y_post = y0, Y_len - y1
            x_pre, x_post = x0, X_len - x1
            pad_widths = ((0, 0), (z_pre, z_post), (y_pre, y_post), (x_pre, x_post))
            return da.pad(block, pad_widths, mode="constant", constant_values=0)

        # ref goes to [r_out_z0:r_out_z1, 0:Yr, 0:Xr]
        ref_canvas = _pad_to_canvas(ref_block, r_out_z0, r_out_z1, 0, Yr, 0, Xr)
        if self.eval_mode:
            ref_canvas[:, Zm:Zm+dzi, :, :] = 0  # no-op to adjust for shift in eval mode?
        # mov goes to [m_out_z0:m_out_z1, m_out_y0:m_out_y1, m_out_x0:m_out_x1]
        mov_canvas = _pad_to_canvas(mov_block, m_out_z0, m_out_z1, m_out_y0, m_out_y1, m_out_x0, m_out_x1)

        # ---- linear crossfade over Z-overlap, lazily (no in-place updates) ----
        # default sum outside overlap
        base_sum = ref_canvas + mov_canvas

        # overlap along Z in ROI coordinates
        ov0 = max(r_out_z0, m_out_z0)
        ov1 = min(r_out_z1, m_out_z1)
        L = ov1 - ov0

        if L > 0 and not self.eval_mode:
            # build broadcastable z-index and weights lazily
            if self.use_gpu:
                z = da.from_array(self.xp.arange(Z_len, dtype=self.xp.int32))[None, :, None, None]
            else:
                z = da.arange(Z_len, dtype=np.int32)[None, :, None, None]
            in_ov = (z >= ov0) & (z < ov1)
            # weight w ranges 0..1 across ov0..ov1-1
            # clamp outside overlap so it doesn't matter
            w = ((z - ov0) / max(L, 1)).astype(np.float32)
            w = da.clip(w, 0.0, 1.0)

            blended = w * ref_canvas + (1.0 - w) * mov_canvas
            fused = da.where(in_ov, blended, base_sum)
        else:
            fused = base_sum

        # if L == 0:
        #     if getattr(self, "seam_sigma_px", None):
        #         # Helper to make a (1, Z_len, 1, 1) z-index lazily
        #         if self.use_gpu:
        #             z_axis = da.from_array(self.xp.arange(Z_len, dtype=self.xp.int32))[None, :, None, None]
        #         else:
        #             z_axis = da.arange(Z_len, dtype=np.int32)[None, :, None, None]
        #
        #         # CASE A: zero overlap (but contiguous) -> feather across a small symmetric window centered at the joins
        #         sz, sy, sx = self.seam_sigma_px
        #         seam_thickness = int(max(1, 3 * sz))
        #         join = (ov0 + ov1) // 2
        #         s = max(0, join - seam_thickness)
        #         e = min(Z_len, join + seam_thickness)
        #         if e > s:
        #             region = fused[:, s:e, :, :]
        #             # Convert CuPy -> NumPy if needed (scipy.ndimage doesn't run on GPU)
        #             # apply 3D Gaussian within this local slab
        #             if self.use_gpu:
        #                 region_np = region.get()
        #                 region_np = gaussian_filter(region_np, sigma=(0, sz, sy, sx))
        #                 region = self.xp.asarray(region_np)
        #             else:
        #                 region = gaussian_filter(region, sigma=(0, sz, sy, sx))
        #
        #             fused[:, s:e] = region

            # CASE B: negative overlap (a small gap) -> fill the gap by interpolating between edge slices
            # Ref ends before mov starts: gap = [r_out_z1, m_out_z0)
            # if r_out_z1 < m_out_z0:
            #     gap_s, gap_e = r_out_z1, m_out_z0
            #     G = gap_e - gap_s
            #     if G > 0:
            #         # Take boundary planes (C,1,Y,X)
            #         # left edge from ref at z=r_out_z1-1, right edge from mov at z=m_out_z0
            #         ref_edge = ref_canvas[:, r_out_z1 - 1:r_out_z1] if r_out_z1 > 0 else da.zeros(
            #             (C_len, 1, Y_len, X_len), dtype=np.float32)
            #         mov_edge = mov_canvas[:, m_out_z0:m_out_z0 + 1] if m_out_z0 < Z_len else da.zeros(
            #             (C_len, 1, Y_len, X_len), dtype=np.float32)
            #         # broadcast linearly across the gap
            #         w_gap = (self.xp.linspace(0.0, 1.0, G, dtype=self.xp.float32)[None, :, None, None])
            #         w_gap = da.from_array(w_gap) if not isinstance(w_gap, da.Array) else w_gap
            #         gap_fill = (1.0 - w_gap) * ref_edge + w_gap * mov_edge
            #         fused = da.concatenate([fused[:, :gap_s], gap_fill, fused[:, gap_e:]], axis=1)
            #
            # # Mirror of Case B: mov ends before ref starts
            # if m_out_z1 < r_out_z0:
            #     gap_s, gap_e = m_out_z1, r_out_z0
            #     G = gap_e - gap_s
            #     if G > 0:
            #         mov_edge = mov_canvas[:, m_out_z1 - 1:m_out_z1] if m_out_z1 > 0 else da.zeros(
            #             (C_len, 1, Y_len, X_len), dtype=np.float32)
            #         ref_edge = ref_canvas[:, r_out_z0:r_out_z0 + 1] if r_out_z0 < Z_len else da.zeros(
            #             (C_len, 1, Y_len, X_len), dtype=np.float32)
            #         w_gap = (self.xp.linspace(0.0, 1.0, G, dtype=self.xp.float32)[None, :, None, None])
            #         w_gap = da.from_array(w_gap) if not isinstance(w_gap, da.Array) else w_gap
            #         gap_fill = (1.0 - w_gap) * mov_edge + w_gap * ref_edge
            #         fused = da.concatenate([fused[:, :gap_s], gap_fill, fused[:, gap_e:]], axis=1)

        # final crop to the requested fused ROI
        # Apply any requested slicing
        fused = fused[:, zf_slice, yf_slice, xf_slice]

        # Trigger compute if Dask array
        if hasattr(fused, "compute"):
            out = fused.compute()
        else:
            out = fused

        # Final cast to dtype and return
        # out = np.asarray(out, dtype=self.dtype)
        return out

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

        # ensure homogeneous backend before stacking
        if self.use_gpu:
            out = self.xp.stack(frames, axis=0)
            out = np.asarray(out.get(), dtype=self.dtype)
        else:
            out = np.stack(frames, axis=0)

        if squeeze_t:
            out = out[0]
        if out.ndim >= 4 and out.shape[-4] == 1:
            out = xp.squeeze(out, axis=-4)
        return out

    def __repr__(self):
        backend = "GPU(CuPy)" if self.use_gpu else "CPU(NumPy)"
        return (f"<VirtualFuseArray shape={self.shape} dtype={self.dtype} "
                f"sides=({self.ref_name}, {self.mov_name}) "
                # f"overlap_z={self.overlap_z} "
                f"interp='{self.interp}' backend={backend}>")
