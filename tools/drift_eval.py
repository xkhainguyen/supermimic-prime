"""Global tracking error of a checkpoint: does the policy drift from the reference in world xy / yaw?

Same protocol as supermimic/evaluate.py (each motion once from its first frame, friction 1.0, no DR or noise).
The reward re-anchors reference bodies to the robot's xy and yaw and the termination checks only height and
tilt, so horizontal drift is never penalised hard. Here, per step while a motion is alive and unfailed:
  root_xy   world xy distance between robot pelvis and reference pelvis
  yaw       |yaw difference| of the pelvis (deg)
  body_glob mean body position error in the world frame (reference not re-anchored)
  body_anch mean body position error with the reference re-anchored as in the reward
Reports, over clips that finish without failing: per-clip mean / final / max root_xy, error vs time since start,
and the share of clips whose final or max root_xy passes 0.25 / 0.5 / 1 m.

Usage: python tools/drift_eval.py --ckpt logs/bones_1gpu_s/ckpt_056000.pt --n_clips 4096 --max_s 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

import supermimic.networks as networks
from supermimic.env import TrackingSim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM
from supermimic.motion import MotionLib
from supermimic.networks import Policy
from supermimic.robot import default_joint_pos_list
from supermimic.rotations import yaw_mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--motion_dir", default="data/bones_seed")
    ap.add_argument("--n_clips", type=int, default=4096)
    ap.add_argument("--max_s", type=float, default=30.0, help="sample only clips up to this length (s); longer ones cost wall time")
    ap.add_argument("--clips", default=None, help="regex on names instead of a random sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda:0"
    kin = G1Kinematics(dev)
    motion = MotionLib.load(a.motion_dir, kin, dev, precompute_bodies=False)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
    if a.clips:
        import re

        pool = [i for i, n in enumerate(motion.names) if re.search(a.clips, n)]
    else:
        pool = (motion.clip_len <= a.max_s * 50).nonzero().flatten().tolist()
    g = torch.Generator().manual_seed(a.seed)
    pick = torch.tensor(pool)[torch.randperm(len(pool), generator=g)[: a.n_clips]]
    sel = pick.to(dev)
    C = len(sel)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    K, Hh = ck["args"]["goal_frames"], ck["args"]["hist"]
    networks.AMP_DTYPE = torch.bfloat16 if ck["args"].get("amp", "none") == "bf16" else None
    pi = Policy(OBS_DIM, 73 * K, hist=Hh).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    env = TrackingSim(C, motion, mdp, eval_mode=True, device=dev, contact_model="paper")
    env.reset(torch.arange(C, device=dev), motion.clip_start[sel].clone())
    clip_end = motion.clip_end[sel]
    T = int(motion.clip_len[sel].max())
    names = ("root_xy", "yaw_deg", "body_glob", "body_anch")
    err = torch.full((len(names), T, C), float("nan"), device=dev)
    failed = torch.zeros(C, dtype=torch.bool, device=dev)
    hist, t = None, 0
    with torch.no_grad():
        while not bool(env.done_eval.all()):
            S = env.state()
            o = mdp.obs(S, env.a_prev, env.xi, env.f)
            hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
            alive = ~env.done_eval.clone()
            out = env.step(pi(hist, motion.goal(env.f, K)))
            failed |= env.failed_eval
            S1 = env.state()
            f = env.f
            p, R = S1[0], S1[1]
            rp, rR = motion.root_p[f], motion.root_R[f]
            bp = kin.forward(*S1)[0]
            mbp = motion.bodies(f)[0]
            dR = yaw_mat(R @ rR.transpose(-1, -2))
            anchor = torch.cat((p[:, :2], rp[:, 2:3]), -1)
            ref_anch = anchor.unsqueeze(1) + (dR.unsqueeze(1) @ (mbp - rp.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
            yaw_r = torch.atan2(rR[:, 1, 0], rR[:, 0, 0])
            yaw_p = torch.atan2(R[:, 1, 0], R[:, 0, 0])
            dyaw = torch.remainder(yaw_p - yaw_r + torch.pi, 2 * torch.pi) - torch.pi
            vals = torch.stack([(p[:, :2] - rp[:, :2]).norm(dim=1), dyaw.abs().rad2deg(),
                                (bp - mbp).norm(dim=-1).mean(-1), (bp - ref_anch).norm(dim=-1).mean(-1)])  # fmt: skip
            ok = alive & ~out["term"]
            err[:, t] = torch.where(ok[None], vals, torch.full_like(vals, float("nan")))
            t += 1
            env.f = torch.minimum(env.f, clip_end - 1)
    err = err[:, :t]
    good = ~failed
    res = {"ckpt": a.ckpt, "clips": C, "failed": int(failed.sum()), "mean_len_s": float(motion.clip_len[sel].float().mean() * 0.02)}
    for k, nm in enumerate(names):
        e = err[k][:, good]  # (T, good)
        res[nm] = {"mean_over_all_steps": float(e.nanmean()), "median_clip_mean": float(torch.nanmedian(e.nanmean(0))),
                   "p95_clip_max": float(torch.quantile(torch.nan_to_num(e, nan=0.0).amax(0), 0.95))}
    xy = err[0][:, good]
    last = torch.stack([xy[~xy[:, c].isnan(), c][-1] for c in range(xy.shape[1])])
    mx = torch.nan_to_num(xy, nan=0.0).amax(0)
    res["root_xy_final_median"], res["root_xy_final_p95"] = float(last.median()), float(torch.quantile(last, 0.95))
    res["share_final_over"] = {f"{th}m": float((last > th).float().mean()) for th in (0.25, 0.5, 1.0)}
    res["share_max_over"] = {f"{th}m": float((mx > th).float().mean()) for th in (0.25, 0.5, 1.0)}
    res["by_time_s"] = {}
    for s in (1, 2, 5, 10, 20, 30):
        if s * 50 <= t:
            res["by_time_s"][s] = {nm: round(float(err[k][s * 50 - 1][good].nanmean()), 4) for k, nm in enumerate(names)}
    # longest clips (>= 10 s) separately: drift accumulates with time
    long = good & (motion.clip_len[sel] >= 500)
    if long.any():
        xyl = err[0][:, long]
        res["long_clips_ge10s"] = {"n": int(long.sum()), "root_xy_mean": float(xyl.nanmean()),
                                   "root_xy_final_median": float(torch.stack([xyl[~xyl[:, c].isnan(), c][-1] for c in range(xyl.shape[1])]).median())}  # fmt: skip
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
