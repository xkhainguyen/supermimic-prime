"""Convert the G1-retargeted BONES-SEED release into one packed 50 Hz motion library (paper App. E).

Input: flat BONES-SEED CSVs (120 fps; Frame, root_translate{X,Y,Z} in cm, root_rotate{X,Y,Z} Euler
in degrees, 29 joint DOFs in degrees). Output (in --out): root_p.npy (F,3) m, root_quat.npy (F,4)
wxyz, q.npy (F,29) rad in G1_JOINT_NAMES order, clip_len.npy, names.txt, filter_report.json.

Filters: (1) name filter, SONIC's released keyword list plus "box" and "look_over_fence";
(2) feasibility filter on MuJoCo forward kinematics at 50 Hz: elevated support, ground
penetration, and floating feet while standing still.

Usage: python tools/prepare_bones.py --src ~/datasets/bones-seed/g1/csv --out data/bones_seed [--check_euler]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

SRC_FPS, DST_FPS = 120.0, 50.0

SONIC_KEYWORDS = (
    "bed, bike, chair, climb, com_up_50cm, sitting, step_on, seat, table, _sit_, sit_, ladder, crutch, _bed_, _ride_, "
    "scooter, stepdown, acrobatics_, box_HSPU, cartwheel, 50cm_box_, on_box, fall_from, handstand_ff_, on_1m, form_box, "
    "off_1m, 230m, jump_over_obstacle_, lift_crate_come_up_, jump_to_shoulder_roll, kozak_dance, stair, handstand, "
    "box_jump, monkey_jump, safety_roll, box_dips, walking_on_edge, push_obstacle"
).split(", ")
EXTRA_KEYWORDS = ["box", "look_over_fence"]


def name_ok(name: str) -> bool:
    n = name.lower()
    return not any(k.lower() in n for k in SONIC_KEYWORDS + EXTRA_KEYWORDS)


def euler_to_quat(e_deg: np.ndarray, order: str) -> np.ndarray:
    """Euler angles (T,3) about x, y, z in degrees -> wxyz quaternions.

    order "xyz": extrinsic x, then y, then z (R = Rz Ry Rx), scipy's lowercase convention, which
    SONIC's converter uses. order "XYZ": intrinsic (R = Rx Ry Rz).
    """
    h = np.deg2rad(e_deg) / 2
    c, s = np.cos(h), np.sin(h)
    qx = np.stack((c[:, 0], s[:, 0], 0 * c[:, 0], 0 * c[:, 0]), -1)
    qy = np.stack((c[:, 1], 0 * c[:, 1], s[:, 1], 0 * c[:, 1]), -1)
    qz = np.stack((c[:, 2], 0 * c[:, 2], 0 * c[:, 2], s[:, 2]), -1)
    return _qmul(_qmul(qz, qy), qx) if order == "xyz" else _qmul(_qmul(qx, qy), qz)


def _qmul(a, b):
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), -1)  # fmt: skip


def _slerp(q0, q1, t):
    dot = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), 0, 1)
    th = np.arccos(dot)
    s = np.sin(th)
    t = t[:, None]
    w0 = np.where(s < 1e-6, 1 - t, np.sin((1 - t) * th) / np.maximum(s, 1e-12))
    w1 = np.where(s < 1e-6, t, np.sin(t * th) / np.maximum(s, 1e-12))
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def load_clip(args):
    path, order = args
    import pandas as pd

    from supermimic.robot import G1_JOINT_NAMES

    d = pd.read_csv(path)
    p = d[["root_translateX", "root_translateY", "root_translateZ"]].to_numpy(np.float64) / 100.0
    quat = euler_to_quat(d[["root_rotateX", "root_rotateY", "root_rotateZ"]].to_numpy(np.float64), order)
    q = np.deg2rad(d[[n + "_dof" for n in G1_JOINT_NAMES]].to_numpy(np.float64))
    T = len(d)
    n = int(np.floor((T - 1) / SRC_FPS * DST_FPS)) + 1
    t = np.arange(n) * SRC_FPS / DST_FPS
    i0 = np.clip(np.floor(t).astype(int), 0, T - 2)
    a = np.clip(t - i0, 0, 1)
    lerp = lambda x: x[i0] * (1 - a[:, None]) + x[i0 + 1] * a[:, None]  # noqa: E731
    return lerp(p).astype(np.float32), _slerp(quat[i0], quat[i0 + 1], a).astype(np.float32), lerp(q).astype(np.float32)


def load_all(files, order, workers):
    with Pool(workers) as pool:
        return pool.map(load_clip, [(f, order) for f in files], chunksize=64)


# ------------------------------------------------------------------------------------------------ feasibility
def feasibility(clips, device):
    """Per-clip flags from FK at 50 Hz (paper App. E thresholds)."""
    import mujoco
    import torch

    from supermimic.kinematics import G1Kinematics
    from supermimic.robot import MJCF_PATH
    from supermimic.rotations import quat_to_mat

    kin = G1Kinematics(device)
    m = mujoco.MjModel.from_xml_path(MJCF_PATH)
    # foot capsules: (body index in kin order, local center, local axis, half length, radius)
    caps = {"left": [], "right": []}
    for g in range(m.ngeom):
        name = m.geom(g).name
        if name.endswith("_collision") and "_foot" in name:
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            side = "left" if name.startswith("left") else "right"
            caps[side].append((kin.body_names.index(m.body(m.geom_bodyid[g]).name), m.geom_pos[g].copy(),
                               R.reshape(3, 3)[:, 2].copy(), m.geom_size[g][1], m.geom_size[g][0]))  # fmt: skip
    lens = np.array([len(c[0]) for c in clips])
    P = torch.from_numpy(np.concatenate([c[0] for c in clips]))
    Q = torch.from_numpy(np.concatenate([c[1] for c in clips]))
    J = torch.from_numpy(np.concatenate([c[2] for c in clips]))
    zmin, foot_h = [], {"left": [], "right": []}
    with torch.no_grad():
        for s in range(0, len(P), 1 << 20):
            p, R, q = P[s : s + (1 << 20)].to(device), quat_to_mat(Q[s : s + (1 << 20)].to(device)), J[s : s + (1 << 20)].to(device)
            z = torch.zeros_like(p)
            bp, bR, _, _ = kin.forward(p, R, z, z, q, torch.zeros_like(q), all_bodies=True)
            zmin.append(bp[..., 2].min(-1).values.cpu())
            for side, lst in caps.items():
                h = []
                for bi, c, ax, hl, r in lst:
                    c_w = bp[:, bi] + bR[:, bi] @ torch.tensor(c, dtype=torch.float32, device=device)
                    a_w = bR[:, bi] @ torch.tensor(ax, dtype=torch.float32, device=device)
                    h.append(torch.minimum(c_w[:, 2] + hl * a_w[:, 2], c_w[:, 2] - hl * a_w[:, 2]) - r)
                foot_h[side].append(torch.stack(h, -1).min(-1).values.cpu())
    zmin = torch.cat(zmin).numpy()
    fl, fr = torch.cat(foot_h["left"]).numpy(), torch.cat(foot_h["right"]).numpy()
    up = quat_to_mat(Q)[:, 2, 2].numpy()
    pz = P[:, 2].numpy()
    starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
    flags = []
    for i, (s, n) in enumerate(zip(starts, lens)):
        sl = slice(s, s + n)
        zm = zmin[sl]
        elevated = _longest_run(zm > 0.20) >= 0.6 * DST_FPS
        penetration = bool((zm < -0.12).any()) or (zm < -0.08).mean() > 0.8
        speed = np.zeros(n)
        qspeed = np.zeros(n)
        if n > 1:
            speed[1:] = np.linalg.norm(np.diff(P[sl].numpy(), axis=0), axis=-1) * DST_FPS
            qspeed[1:] = np.abs(np.diff(J[sl].numpy(), axis=0)).max(-1) * DST_FPS
        standing_still = (up[sl] > 0.9) & (pz[sl] > 0.65) & (speed < 0.15) & (qspeed < 1.0)
        floating = _longest_run((fl[sl] > 0.02) & (fr[sl] > 0.02) & standing_still) >= 0.5 * DST_FPS
        flags.append((elevated, penetration, floating))
    return np.array(flags), zmin, starts, lens


def _longest_run(b: np.ndarray) -> int:
    if not b.any():
        return 0
    x = np.concatenate(([0], b.astype(np.int8), [0]))
    d = np.diff(x)
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expanduser("~/datasets/bones-seed/g1/csv"))
    ap.add_argument("--out", default="data/bones_seed")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--order", default="xyz", choices=["xyz", "XYZ"])
    ap.add_argument("--check_euler", action="store_true", help="compare Euler conventions on acrobatic clips and exit")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.src, "*", "*.csv")))
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    print(f"raw clips: {len(files)}", flush=True)

    if a.check_euler:
        sel = [f for f, n in zip(files, names) if name_ok(n) and any(k in n for k in ("flip", "roll", "lying", "crawl", "get_up"))][:400]
        for order in ("xyz", "XYZ"):
            clips = load_all(sel, order, a.workers)
            flags, zmin, starts, lens = feasibility(clips, a.device)
            print(f"{order}: {len(sel)} clips, frames z_min<-0.05: {(zmin < -0.05).mean():.4f}, z_min>0.15: {(zmin > 0.15).mean():.4f}, "
                  f"flagged elevated/penetration/floating: {flags.sum(0).tolist()}", flush=True)  # fmt: skip
        return

    keep = [i for i, n in enumerate(names) if name_ok(n)]
    print(f"after name filter: {len(keep)}", flush=True)
    t0 = time.time()
    clips = load_all([files[i] for i in keep], a.order, a.workers)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    flags, _, _, lens = feasibility(clips, a.device)
    ok = ~flags.any(1)
    print(f"feasibility: elevated {int(flags[:, 0].sum())}, penetration {int(flags[:, 1].sum())}, floating {int(flags[:, 2].sum())}, "
          f"removed {int((~ok).sum())}, kept {int(ok.sum())}", flush=True)  # fmt: skip
    os.makedirs(a.out, exist_ok=True)
    kept = [clips[i] for i in np.flatnonzero(ok)]
    np.save(os.path.join(a.out, "root_p.npy"), np.concatenate([c[0] for c in kept]))
    np.save(os.path.join(a.out, "root_quat.npy"), np.concatenate([c[1] for c in kept]))
    np.save(os.path.join(a.out, "q.npy"), np.concatenate([c[2] for c in kept]))
    np.save(os.path.join(a.out, "clip_len.npy"), lens[ok])
    kept_names = [names[keep[i]] for i in np.flatnonzero(ok)]
    open(os.path.join(a.out, "names.txt"), "w").write("\n".join(kept_names) + "\n")
    removed = {names[keep[i]]: [k for k, f in zip(("elevated", "penetration", "floating"), flags[i]) if f] for i in np.flatnonzero(~ok)}
    report = {"raw": len(files), "after_name_filter": len(keep), "kept": int(ok.sum()), "hours": float(lens[ok].sum() / DST_FPS / 3600),
              "euler_order": a.order, "removed_feasibility": removed}  # fmt: skip
    json.dump(report, open(os.path.join(a.out, "filter_report.json"), "w"), indent=1)
    print(f"kept {report['kept']} clips, {report['hours']:.2f} h -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
