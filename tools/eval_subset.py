"""Evaluate a checkpoint on a subset of motions (name regex), with per-motion failure details.

Same protocol as supermimic/evaluate.py (each motion once from its first frame, friction 1.0, no DR or
noise, deterministic policy, failure = training termination), plus: the fraction of the clip reached
before failing, and which termination fired first (root height error > 0.25 m, tilt error > 0.8).

Usage: python tools/eval_subset.py --ckpt logs/bones_2gpu_c/ckpt_024000.pt --clips "flip|side_roll" --contact_model paper
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

import supermimic.networks as networks
from supermimic.env import TrackingSim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MAX_STEP_REWARD, MDP, OBS_DIM
from supermimic.motion import MotionLib
from supermimic.networks import Policy
from supermimic.robot import default_joint_pos_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--motion_dir", default="data/bones_seed")
    ap.add_argument("--clips", required=True, help="regex on motion names")
    ap.add_argument("--contact_model", default="paper", choices=["paper", "legacy"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda:0"
    kin = G1Kinematics(dev)
    motion = MotionLib.load(a.motion_dir, kin, dev, precompute_bodies=False)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
    sel = torch.tensor([i for i, n in enumerate(motion.names) if re.search(a.clips, n)], device=dev)
    C = len(sel)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    K, Hh = ck["args"]["goal_frames"], ck["args"]["hist"]
    networks.AMP_DTYPE = torch.bfloat16 if ck["args"].get("amp", "none") == "bf16" else None
    pi = Policy(OBS_DIM, 73 * K, hist=Hh).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    env = TrackingSim(C, motion, mdp, eval_mode=True, device=dev, contact_model=a.contact_model)
    ids = torch.arange(C, device=dev)
    env.reset(ids, motion.clip_start[sel].clone())
    clip_end = motion.clip_end[sel]
    ret = torch.zeros(C, device=dev)
    fail_frame = torch.full((C,), -1, dtype=torch.long, device=dev)
    fail_kind = torch.zeros(C, dtype=torch.long, device=dev)  # 1 height, 2 tilt, 3 both
    hist = None
    with torch.no_grad():
        while not bool(env.done_eval.all()):
            S = env.state()
            o = mdp.obs(S, env.a_prev, env.xi, env.f)
            hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
            alive = ~env.done_eval.clone()
            out = env.step(pi(hist, motion.goal(env.f, K)))
            ret += torch.where(alive, out["r"], 0.0)
            S1 = env.state()
            rp, rR = motion.root_p[env.f], motion.root_R[env.f]
            h = (rp[:, 2] - S1[0][:, 2]).abs() > mdp.cfg.term_height
            t = (rR[:, 2, 2] - S1[1][:, 2, 2]).abs() > mdp.cfg.term_tilt
            new = out["term"] & alive
            fail_frame = torch.where(new, env.f - motion.clip_start[sel], fail_frame)
            fail_kind = torch.where(new, h.long() + 2 * t.long(), fail_kind)
            env.f = torch.minimum(env.f, clip_end - 1)
    norm = ret / ((motion.clip_len[sel] - 1).float() * MAX_STEP_REWARD)
    kinds = {0: "-", 1: "height", 2: "tilt", 3: "height+tilt"}
    rows = []
    for i in range(C):
        L = int(motion.clip_len[sel[i]])
        ff = int(fail_frame[i])
        rows.append({"name": motion.names[int(sel[i])], "len_s": L * 0.02, "failed": ff >= 0, "fail_at": ff / L if ff >= 0 else None,
                     "fail_s": ff * 0.02 if ff >= 0 else None, "kind": kinds[int(fail_kind[i])], "return": round(float(norm[i]), 4)})  # fmt: skip
    n_fail = sum(r["failed"] for r in rows)
    print(f"{a.ckpt} | contact {a.contact_model} | /{a.clips}/: {C} motions, failed {n_fail}, mean return {float(norm.mean()):.3f}")
    for r in rows:
        print(f"  {r['name']:42s} {r['len_s']:5.1f}s  " + (f"FAIL at {r['fail_s']:5.2f}s ({r['fail_at']:.0%}) {r['kind']:11s}" if r["failed"] else "ok" + " " * 29) + f" ret {r['return']:.3f}")
    if a.out:
        json.dump({"ckpt": a.ckpt, "contact_model": a.contact_model, "clips": a.clips, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
