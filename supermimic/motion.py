"""Reference motion library at 50 Hz with FK of tracked bodies.

Two sources: LAFAN1 G1 retarget CSVs (30 fps, resampled here) and a packed 50 Hz library
(tools/prepare_bones.py; root_p, root_quat, q, clip_len). Tracked-body poses and velocities are
precomputed by default; large libraries store body orientations as quaternions (BONES-SEED, 47.3M
frames: ~34 GB instead of ~55 GB with matrices). ``precompute_bodies=False`` computes them by FK on
demand instead (13% slower training, used by evaluation jobs).
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

from .kinematics import G1Kinematics
from .rotations import mat_to_6d, mat_to_quat, quat_slerp, quat_to_mat, so3_log, yaw_mat

CONTROL_DT = 0.02
BIN_FRAMES = 50  # 1 s bins for return-based reset sampling
BODY_MATRICES_MAX_FRAMES = 2_000_000
CHUNK = 1 << 20


class MotionLib:
    def __init__(self, names: list[str], lens: list[int], root_p: torch.Tensor, root_quat: torch.Tensor, q: torch.Tensor,
                 kin: G1Kinematics, device: str, precompute_bodies: bool = True):  # fmt: skip
        """root_p (F,3), root_quat (F,4) wxyz, q (F,29) at 50 Hz, clips concatenated in ``names`` order."""
        self.device, self.kin, self.names = device, kin, names
        self.clip_len = torch.as_tensor(lens, device=device)
        self.clip_start = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.long, device=device), self.clip_len[:-1])), 0)
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.num_frames = int(self.clip_len.sum())
        self.clip_id = torch.repeat_interleave(torch.arange(len(lens), device=device), self.clip_len)
        F_ = self.num_frames
        self.root_p, self.q = root_p.to(device), q.to(device)
        self.root_R = torch.empty(F_, 3, 3, device=device)
        for s in range(0, F_, CHUNK):
            self.root_R[s : s + CHUNK] = quat_to_mat(root_quat[s : s + CHUNK].to(device))
        # finite differences inside each clip: central, one-sided at the clip boundaries
        idx = torch.arange(F_, device=device)
        first = idx == self.clip_start[self.clip_id]
        last = idx == self.clip_end[self.clip_id]
        nxt = torch.where(last, idx, idx + 1)
        prv = torch.where(first, idx, idx - 1)
        dt = ((nxt - prv).float() * CONTROL_DT).clamp(min=CONTROL_DT)
        self.root_v = (self.root_p[nxt] - self.root_p[prv]) / dt[:, None]
        self.qd = (self.q[nxt] - self.q[prv]) / dt[:, None]
        self.root_w = torch.empty(F_, 3, device=device)
        for s in range(0, F_, CHUNK):
            n_, p_ = nxt[s : s + CHUNK], prv[s : s + CHUNK]
            self.root_w[s : s + CHUNK] = so3_log(self.root_R[n_] @ self.root_R[p_].transpose(-1, -2)) / dt[s : s + CHUNK, None]
        del idx, first, last, nxt, prv, dt
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (Rt @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (Rt @ self.root_w.unsqueeze(-1)).squeeze(-1)
        self.root_gz = self.root_R[:, 2, 2]
        self._bodies, self._body_quat = None, False
        if precompute_bodies:
            nb_ = len(kin.tracked_idx)
            self._body_quat = F_ > BODY_MATRICES_MAX_FRAMES
            rot_shape = (F_, nb_, 4) if self._body_quat else (F_, nb_, 3, 3)
            bp, bR = torch.empty(F_, nb_, 3, device=device), torch.empty(rot_shape, device=device)
            bv, bw = torch.empty(F_, nb_, 3, device=device), torch.empty(F_, nb_, 3, device=device)
            with torch.no_grad():
                for s in range(0, F_, 65536):
                    sl = slice(s, min(s + 65536, F_))
                    p_, R_, v_, w_ = self._fk(torch.arange(sl.start, sl.stop, device=device))
                    bp[sl], bv[sl], bw[sl] = p_, v_, w_
                    bR[sl] = mat_to_quat(R_) if self._body_quat else R_
            self._bodies = (bp, bR, bv, bw)
        # 1 s bins; a clip's leftover frames (< 1 s) join its last bin, so no bin is only a few frames long
        nb = torch.clamp(torch.div(self.clip_len, BIN_FRAMES, rounding_mode="floor"), min=1)
        self.bin_clip = torch.repeat_interleave(torch.arange(len(lens), device=device), nb)
        bin_off = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.long, device=device), nb[:-1])), 0)
        local_bin = torch.arange(len(self.bin_clip), device=device) - bin_off[self.bin_clip]
        self.bin_start = self.clip_start[self.bin_clip] + local_bin * BIN_FRAMES
        last = local_bin == nb[self.bin_clip] - 1
        self.bin_end = torch.where(last, self.clip_end[self.bin_clip] + 1, self.bin_start + BIN_FRAMES)  # exclusive
        self.num_bins = len(self.bin_start)
        local_frame = torch.arange(F_, device=device) - self.clip_start[self.clip_id]
        self.frame_bin = bin_off[self.clip_id] + torch.minimum(torch.div(local_frame, BIN_FRAMES, rounding_mode="floor"), nb[self.clip_id] - 1)

    def _fk(self, f: torch.Tensor):
        return self.kin.forward(self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f])

    def bodies(self, f: torch.Tensor):
        """Tracked-body world positions, rotations, linear and angular velocities of reference frames f."""
        if self._bodies is None:
            return self._fk(f)
        bp, bR, bv, bw = (x[f] for x in self._bodies)
        return bp, quat_to_mat(bR) if self._body_quat else bR, bv, bw

    # ------------------------------------------------------------------ sources
    @staticmethod
    def from_dir(path: str, kin: G1Kinematics, device: str, src_fps: float = 30.0, precompute_bodies: bool = True) -> MotionLib:
        """LAFAN1 G1 retarget CSVs (root pos, root quat xyzw, 29 joints) resampled to 50 Hz."""
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        ps, qs, js, lens = [], [], [], []
        for f in files:
            raw = torch.tensor(np.loadtxt(f, delimiter=","), dtype=torch.float32)
            p, q_xyzw, j = raw[:, :3], raw[:, 3:7], raw[:, 7:36]
            quat = torch.cat((q_xyzw[:, 3:4], q_xyzw[:, :3]), -1)
            T_src = raw.shape[0]
            n = int(np.floor((T_src - 1) / src_fps / CONTROL_DT)) + 1
            t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * src_fps
            i0 = torch.clamp(t.floor().long(), max=T_src - 2)
            a = (t - i0.float()).clamp(0, 1)
            ps.append(p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None])
            js.append(j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None])
            qs.append(quat_slerp(quat[i0], quat[i0 + 1], a))
            lens.append(n)
        return MotionLib(names, lens, torch.cat(ps), torch.cat(qs), torch.cat(js), kin, device, precompute_bodies)

    @staticmethod
    def from_packed(path: str, kin: G1Kinematics, device: str, precompute_bodies: bool = True) -> MotionLib:
        """Packed 50 Hz library written by tools/prepare_bones.py."""
        load = lambda n: torch.from_numpy(np.load(os.path.join(path, n)))  # noqa: E731
        names = open(os.path.join(path, "names.txt")).read().splitlines()
        lens = np.load(os.path.join(path, "clip_len.npy")).tolist()
        return MotionLib(names, lens, load("root_p.npy"), load("root_quat.npy"), load("q.npy"), kin, device, precompute_bodies)

    @staticmethod
    def load(path: str, kin: G1Kinematics, device: str, precompute_bodies: bool = True) -> MotionLib:
        if os.path.exists(os.path.join(path, "clip_len.npy")):
            return MotionLib.from_packed(path, kin, device, precompute_bodies)
        return MotionLib.from_dir(path, kin, device, precompute_bodies=precompute_bodies)

    # ------------------------------------------------------------------ queries
    def future_frames(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Frames f+1..f+K clamped to the clip's last frame, shape (B, K)."""
        end = self.clip_end[self.clip_id[f]]
        fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        return torch.minimum(fut, end.unsqueeze(-1))

    def goal(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Goal features from the reference only: K future frames relative to the current reference frame."""
        fut = self.future_frames(f, K)  # (B,K)
        H = yaw_mat(self.root_R[f])  # (B,3,3)
        Ht = H.transpose(-1, -2).unsqueeze(1)
        dp = (Ht @ (self.root_p[fut] - self.root_p[f].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rot = mat_to_6d(Ht @ self.root_R[fut])
        feats = torch.cat(
            (dp, rot, self.root_v_local[fut], self.root_w_local[fut], self.q[fut], self.qd[fut] * 0.1), -1
        )  # (B,K,73)
        return feats.reshape(f.shape[0], -1)

    def ref_state(self, f: torch.Tensor):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]
