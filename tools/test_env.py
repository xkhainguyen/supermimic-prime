"""Check FK vs simulator, state round-trip, reference playback, and simulator throughput."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--bench_envs", type=int, default=4096)
args = parser.parse_args()

import torch

from supermimic.env import TrackingSim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP
from supermimic.motion import MotionLib
from supermimic.robot import TRACKED_BODIES, default_joint_pos_list
from supermimic.rotations import quat_to_mat, rot_angle_sq

dev = "cuda:0"
kin = G1Kinematics(dev)
t0 = time.time()
motion = MotionLib.from_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "lafan1_g1"), kin, dev)
print(f"motion: {len(motion.names)} clips, {motion.num_frames} frames ({motion.num_frames * 0.02 / 3600:.2f} h), bins {motion.num_bins}, load {time.time() - t0:.1f}s")
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
env = TrackingSim(args.num_envs, motion, mdp, eval_mode=True, device=dev)
print("alpha", [round(x, 3) for x in env.alpha.tolist()])
frames = motion.clip_start[torch.arange(args.num_envs, device=dev) % len(motion.names)] + 100
env.reset(torch.arange(args.num_envs, device=dev), frames)
env.sim.forward()
S = env.state()
ref = motion.ref_state(frames)
for n, a, b in zip(["p", "R", "v", "w", "q", "qd"], S, ref):
    print(f"roundtrip {n}: max err {(a - b).abs().max().item():.2e}")
# FK vs sim body frames (link origins)
d = env.robot.data
ids = [env.robot.body_names.index(b) for b in TRACKED_BODIES]
pose = d.body_link_pose_w[:, ids]
vel = d.body_link_vel_w[:, ids]
bp, bR, bv, bw = kin.forward(*S)
print("FK pos err", (bp - pose[..., :3]).norm(dim=-1).max().item())
print("FK rot err", rot_angle_sq(bR, quat_to_mat(pose[..., 3:7])).sqrt().max().item())
print("FK linvel err", (bv - vel[..., :3]).norm(dim=-1).max().item(), " angvel err", (bw - vel[..., 3:]).norm(dim=-1).max().item())
print("ref body vs FK(sim) pos err", (motion.body_p[frames] - bp).norm(dim=-1).max().item())
# open-loop reference playback via joint targets
for t in range(150):
    f1 = env.f + 1
    a = (motion.q[f1] - env.default_q) / env.alpha
    out = env.step(a)
    if t % 25 == 0:
        print(t, "norm reward", out["r"].mean().item() / 0.1, "term frac", env.failed_eval.float().mean().item(), "e_body_pos", out["terms"]["e_body_pos"].mean().item())
del env

# throughput at training scale (warm: discard the first 50 steps)
env = TrackingSim(args.bench_envs, motion, mdp, device=dev)
a = torch.zeros(args.bench_envs, 29, device=dev)
for _ in range(50):
    env.step(a + 0.1 * torch.randn_like(a))
torch.cuda.synchronize()
t0 = time.time()
n = 200
for _ in range(n):
    env.step(a + 0.1 * torch.randn_like(a))
torch.cuda.synchronize()
dt = time.time() - t0
print(f"sim throughput @ {args.bench_envs} envs: {n * args.bench_envs / dt / 1e3:.0f}k control steps/s ({1e3 * dt / n:.1f} ms/step, incl. reward/term/reset)")
