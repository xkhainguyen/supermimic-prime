"""Checkpoint evaluation (paper Appendix F.1) on all motions of the library.

Each motion plays once from its first frame with friction 1.0, no domain randomization, no
observation or action noise. A motion fails if the rollout triggers the training termination.
Normalized return is the undiscounted reward sum divided by its maximum attainable value.

Run as a watcher: ``python supermimic/evaluate.py --run_dir logs/<run> --watch``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--watch", action="store_true")
parser.add_argument("--ckpt", type=str, default=None, help="evaluate one checkpoint")
parser.add_argument("--record", type=str, default=None, help="save trajectories (npz) for rendering")
parser.add_argument("--motion_dir", type=str, default=os.path.join(REPO, "data", "lafan1_g1"))
parser.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
parser.add_argument("--eval_envs", type=int, default=16384, help="parallel motions; larger libraries run in length-sorted chunks")
args = parser.parse_args()

import numpy as np
import torch

import supermimic.networks as networks
from supermimic.env import TrackingSim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MAX_STEP_REWARD, OBS_DIM, MDP
from supermimic.motion import MotionLib
from supermimic.networks import Policy
from supermimic.robot import default_joint_pos_list

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"
kin = G1Kinematics(dev)
motion = MotionLib.load(args.motion_dir, kin, dev, precompute_bodies=False)  # eval time is not counted; fits 48 GB GPUs
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
C = len(motion.names)


@torch.no_grad()
def evaluate(env: TrackingSim, ck: dict, record: str | None = None) -> dict:
    a_ = ck["args"]
    K, Hh = a_["goal_frames"], a_["hist"]
    networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
    pi = Policy(OBS_DIM, 73 * K, hist=Hh).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    N = env.N
    assert record is None or C <= N, "--record needs every motion in flight at once"
    ret = torch.zeros(C, device=dev)
    failed = torch.zeros(C, dtype=torch.bool, device=dev)
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
    # work queue, longest motions first: an env that finishes a motion starts the next one
    order = torch.argsort(motion.clip_len, descending=True)
    n0 = min(N, C)
    slot = torch.full((N,), -1, dtype=torch.long, device=dev)  # motion index per env, -1 = idle
    slot[:n0] = order[:n0]
    nxt = n0
    replay = order[torch.arange(N, device=dev) % n0]  # idle envs replay a motion and are ignored
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(torch.arange(N, device=dev), motion.clip_start[replay])
    env.done_eval[n0:] = True
    clip_end = motion.clip_end[replay].clone()
    r_sum = torch.zeros(N, device=dev)
    fresh = torch.ones(N, dtype=torch.bool, device=dev)
    hist = None
    while True:
        S = env.state()
        o = mdp.obs(S, env.a_prev, env.xi, env.f)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        hist[fresh] = o[fresh].unsqueeze(1).expand(-1, Hh, -1)
        fresh[:] = False
        a = pi(hist, motion.goal(env.f, K))
        alive = ~env.done_eval.clone()
        if record:
            from supermimic.rotations import mat_to_quat

            traj["root_p"].append(S[0][:n0].cpu()), traj["root_quat"].append(mat_to_quat(S[1])[:n0].cpu())
            traj["q"].append(S[4][:n0].cpu()), traj["f"].append(env.f[:n0].cpu()), traj["alive"].append(alive[:n0].cpu())
        out = env.step(a)
        r_sum += torch.where(alive, out["r"], 0.0)
        env.f = torch.minimum(env.f, clip_end - 1)  # finished motions keep stepping; keep f+1 in range
        finished = env.done_eval & (slot >= 0)
        if record is None and bool(finished.any()):
            ids = finished.nonzero(as_tuple=False).squeeze(-1)
            ret[slot[ids]] = r_sum[ids]
            failed[slot[ids]] = env.failed_eval[ids]
            k = min(len(ids), C - nxt)
            if k > 0:
                new_ids, new = ids[:k], order[nxt : nxt + k]
                nxt += k
                slot[new_ids] = new
                env.done_eval[new_ids] = False
                env.failed_eval[new_ids] = False
                env.reset(new_ids, motion.clip_start[new])
                clip_end[new_ids] = motion.clip_end[new]
                r_sum[new_ids] = 0.0
                fresh[new_ids] = True
            slot[ids[k:]] = -1
        if bool(env.done_eval.all()):
            break
    if record is None:
        live = slot >= 0
        ret[slot[live]] = r_sum[live]
        failed[slot[live]] = env.failed_eval[live]
    else:
        ret[order[:n0]] = r_sum[:n0]
        failed[order[:n0]] = env.failed_eval[:n0]
    if record:  # back to library order
        inv = torch.argsort(order[:n0]).cpu()
        traj = {k: [x[inv] for x in v] for k, v in traj.items()}
    norm = ret / ((motion.clip_len - 1).float() * MAX_STEP_REWARD)
    res = {
        "norm_return": norm.mean().item(),
        "norm_return_framew": (ret.sum() / ((motion.clip_len - 1).sum() * MAX_STEP_REWARD)).item(),
        "n_failed": int(failed.sum().item()),
        "failed": [motion.names[i] for i in failed.nonzero().flatten().tolist()],
        "per_clip": {motion.names[i]: round(norm[i].item(), 4) for i in range(C)},
    }
    if record:
        np.savez_compressed(record, names=np.array(motion.names), **{k: torch.stack(v).numpy() for k, v in traj.items()},
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy())  # fmt: skip
    return res


env = TrackingSim(min(C, args.eval_envs), motion, mdp, eval_mode=True, device=dev)
if args.ckpt:
    ck = torch.load(args.ckpt, map_location=dev)
    res = evaluate(env, ck, args.record)
    print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))
    sys.exit(0)
out_csv = os.path.join(args.run_dir, "eval.csv")
done = set()
if os.path.exists(out_csv):
    done = {int(l.split(",")[0]) for l in open(out_csv).read().strip().splitlines()[1:]}
else:
    open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
while True:
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))
    todo = [c for c in ckpts if int(os.path.basename(c)[5:11]) not in done]
    todo = [c for c in todo if (int(os.path.basename(c)[5:11]) // 2000) % args.every == 0]
    for c in todo:
        it = int(os.path.basename(c)[5:11])
        ck = torch.load(c, map_location=dev)
        t0 = time.time()
        res = evaluate(env, ck)
        line = f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(res['failed'][:50])}\n"
        open(out_csv, "a").write(line)
        json.dump(res, open(c.replace(".pt", "_eval.json"), "w"), indent=1)
        print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C} ({time.time() - t0:.0f}s)", flush=True)
        done.add(it)
    if not args.watch:
        break
    time.sleep(60)
