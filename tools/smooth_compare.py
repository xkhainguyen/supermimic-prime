"""Is the world-model policy gradient a randomized-smoothing gradient of the simulator?

Objective as in tools/apg_compare.py: J = sum_{t<H} gamma^t c_t r_{t+1} + gamma^H c_H V(s_H), closed loop with
the trained policy, analytic reward / soft termination / critic on each system's states, no noise.
For a perturbation u of the action sequence (added to the policy's output at every step, H x 29):

  WM:        g_wm      = dJ_wm/du at u = 0 (autograd, BPTT)
  simulator: g_sim(s)  = E_{u ~ N(0, s^2 I)} [ dJ_sim/du (u) ]  (smoothed gradient), estimated by the mean of
             central finite differences (step eps) at n_samples random base perturbations u.

Reported per sigma: cosine and norm ratio WM vs simulator for dJ/du and for dJ/dtheta, and a split-half
cosine of the simulator estimate (estimator noise).

Usage: python tools/smooth_compare.py --ckpt logs/lafan1_main/last.pt --sigmas 0 0.01 0.03 0.1 0.3
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
from supermimic.mdp import CRITIC_EXTRA, INV_DIM, MDP, OBS_DIM, normalize_xi
from supermimic.motion import MotionLib
from supermimic.networks import Critic, Policy, WorldModel
from supermimic.robot import default_joint_pos_list
from supermimic.rotations import quat_to_mat

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="logs/lafan1_main/last.pt")
ap.add_argument("--motion_dir", default="data/lafan1_g1")
ap.add_argument("--contact_model", default="legacy", choices=["paper", "legacy"])
ap.add_argument("--clips", nargs="+", default=["walk1_subject1", "run1_subject2", "dance2_subject1", "fallAndGetUp1_subject1"])
ap.add_argument("--start_s", type=float, default=3.0)
ap.add_argument("--H", type=int, default=32)
ap.add_argument("--eps", type=float, default=1e-3)
ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.1, 0.3])
ap.add_argument("--n_samples", type=int, default=8)
ap.add_argument("--chunk", type=int, default=16384)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/smooth_compare.json")
a = ap.parse_args()

torch.manual_seed(a.seed)
dev = "cuda:0"
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
cargs = ck["args"]
networks.AMP_DTYPE = None
torch.backends.cuda.matmul.allow_tf32 = False
K, Hh, H, gamma = cargs["goal_frames"], cargs["hist"], a.H, cargs["gamma"]
kin = G1Kinematics(dev)
motion = MotionLib.load(a.motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
wm = WorldModel().to(dev)
wm.load_state_dict(ck["wm"])
pi = Policy(OBS_DIM, 73 * K, hist=Hh).to(dev)
pi.load_state_dict(ck["pi"])
critic = Critic(INV_DIM + CRITIC_EXTRA + 73 * K).to(dev)
critic.load_state_dict(ck["critic"])
for m_ in (wm, pi, critic):
    m_.eval()
for p in list(wm.parameters()) + list(critic.parameters()):
    p.requires_grad_(False)
params = list(pi.parameters())
M = len(a.clips)
clip_ids = torch.tensor([motion.names.index(c) for c in a.clips], device=dev)


def objective(S_list, f_list, A_list, Ap_list, xi_n, f_H):
    B = S_list[0][0].shape[0]
    flat = lambda xs: torch.cat(xs, 0)  # noqa: E731
    S_all = tuple(flat([St[i] for St in S_list]) for i in range(6))
    r = mdp.reward(S_all, flat(f_list), flat(A_list), flat(Ap_list)).view(H, B)
    d = mdp.soft_termination(S_all, flat(f_list)).view(H, B)
    alive = torch.cumprod(1 - d, 0)
    c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
    disc = gamma ** torch.arange(H, device=dev, dtype=torch.float32).unsqueeze(-1)
    V = critic(torch.cat((mdp.critic_features(S_list[-1], f_H, xi_n), motion.goal(f_H, K)), -1))
    return (disc * c_t * r).sum(0) + gamma**H * alive[-1] * V


# ------------------------------------------------------------------------------------------------ start states
base = TrackingSim(M, motion, mdp, eval_mode=True, device=dev, contact_model=a.contact_model)
base.reset(torch.arange(M, device=dev), motion.clip_start[clip_ids])
hist = None
with torch.no_grad():
    for _ in range(int(a.start_s / 0.02)):
        o = mdp.obs(base.state(), base.a_prev, base.xi, base.f)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        base.step(pi(hist, motion.goal(base.f, K)))
    hist = torch.cat((hist[:, 1:], mdp.obs(base.state(), base.a_prev, base.xi, base.f).unsqueeze(1)), 1)
qpos0, qvel0 = base.sim.data.qpos.clone(), base.sim.data.qvel.clone()
f0, a_prev0, xi = base.f.clone(), base.a_prev.clone(), base.xi.clone()
xi_n = normalize_xi(xi)
q_adr, v_adr = base.q_adr, base.v_adr
del base

# ------------------------------------------------------------------------------------------------ simulator, chunked
sim = TrackingSim(a.chunk, motion, mdp, eval_mode=True, device=dev, contact_model=a.contact_model)


@torch.no_grad()
def sim_J(start: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """J for each (start index, action-sequence perturbation H x 29); batched over worlds in chunks."""
    out = []
    for c0 in range(0, len(start), a.chunk):
        st, dl = start[c0 : c0 + a.chunk], delta[c0 : c0 + a.chunk]
        n = len(st)
        pad = a.chunk - n
        if pad:
            st, dl = torch.cat((st, st[:1].expand(pad))), torch.cat((dl, dl[:1].expand(pad, -1, -1)))
        sim.sim.reset()
        sim.robot.reset()
        sim.sim.data.qpos[:] = qpos0[st]
        sim.sim.data.qvel[:] = qvel0[st]
        sim.f, sim.a_prev = f0[st].clone(), a_prev0[st].clone()
        sim.done_eval[:] = False
        sim.failed_eval[:] = False
        h = hist[st].clone()
        S_list, f_list, A_list, Ap_list = [], [], [], []
        for t in range(H):
            act = pi(h, motion.goal(sim.f, K)) + dl[:, t]
            Ap_list.append(sim.a_prev.clone()), A_list.append(act)
            sim.step(act)
            S = sim.state()
            S_list.append(S), f_list.append(sim.f.clone())
            h = torch.cat((h[:, 1:], mdp.obs(S, act, sim.xi, sim.f).unsqueeze(1)), 1)
        out.append(objective(S_list, f_list, A_list, Ap_list, xi_n[st], f_list[-1])[:n])
    return torch.cat(out)


def sim_trajs(ms, us):
    """Simulator trajectories (histories, frames per step) for a batch of (start, perturbation u), for da/dtheta."""
    B = len(ms)
    st = torch.tensor(ms, device=dev)
    U = torch.stack(us)
    st_p, U_p = torch.cat((st, st[:1].expand(a.chunk - B))), torch.cat((U, U[:1].expand(a.chunk - B, -1, -1)))
    with torch.no_grad():
        sim.sim.reset()
        sim.robot.reset()
        sim.sim.data.qpos[:] = qpos0[st_p]
        sim.sim.data.qvel[:] = qvel0[st_p]
        sim.f, sim.a_prev = f0[st_p].clone(), a_prev0[st_p].clone()
        sim.done_eval[:] = False
        h = hist[st_p].clone()
        hs, fs = [], []
        for t in range(H):
            hs.append(h[:B].clone()), fs.append(sim.f[:B].clone())
            act = pi(h, motion.goal(sim.f, K)) + U_p[:, t]
            sim.step(act)
            h = torch.cat((h[:, 1:], mdp.obs(sim.state(), act, sim.xi, sim.f).unsqueeze(1)), 1)
    return hs, fs


def flat_grad(gs):
    return torch.cat([g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=dev) for g, p in zip(gs, params)])


def theta_grad(hs, fs, i, G):
    acts = torch.stack([pi(hs[t][i : i + 1], motion.goal(fs[t][i : i + 1], K))[0] for t in range(H)])
    return flat_grad(torch.autograd.grad(acts, params, G))


# one-hot finite-difference perturbations (t, j, sign)
D = torch.zeros(2 * H * 29, H, 29, device=dev)
for k in range(H * 29):
    D[2 * k].view(-1)[k] = a.eps
    D[2 * k + 1].view(-1)[k] = -a.eps


def fd_grad_batch(ms, us):
    """Central-difference dJ/du for a list of (start, base perturbation u)."""
    start = torch.cat([torch.full((len(D),), m, device=dev) for m in ms])
    delta = torch.cat([u.unsqueeze(0) + D for u in us])
    J = sim_J(start, delta).view(len(ms), H * 29, 2)
    return ((J[..., 0] - J[..., 1]) / (2 * a.eps)).view(len(ms), H, 29)


# ------------------------------------------------------------------------------------------------ world model
R0 = quat_to_mat(qpos0[:, 3:7])
S0 = (qpos0[:, 0:3], R0, qvel0[:, 0:3], (R0 @ qvel0[:, 3:6].unsqueeze(-1)).squeeze(-1), qpos0[:, q_adr], qvel0[:, v_adr])
G_wm, g_wm = [], []
for m in range(M):
    u = torch.zeros(H, 1, 29, device=dev, requires_grad=True)
    S = tuple(x[m : m + 1] for x in S0)
    hm, ap, f = hist[m : m + 1], a_prev0[m : m + 1], f0[m : m + 1]
    Sl, fl, Al, Apl = [], [], [], []
    for t in range(H):
        act = pi(hm, motion.goal(f, K)) + u[t]
        Apl.append(ap), Al.append(act)
        S = wm.step(S, act, xi_n[m : m + 1])
        f = f + 1
        Sl.append(S), fl.append(f)
        hm = torch.cat((hm[:, 1:], mdp.obs(S, act, xi[m : m + 1], f).unsqueeze(1)), 1)
        ap = act
    J = objective(Sl, fl, Al, Apl, xi_n[m : m + 1], fl[-1])[0]
    gp = torch.autograd.grad(J, params + [u])
    g_wm.append(flat_grad(gp[:-1]))
    G_wm.append(gp[-1][:, 0].detach())


def cos(x, y):
    return float((x * y).sum() / (x.norm() * y.norm() + 1e-20))


# ------------------------------------------------------------------------------------------------ smoothed simulator gradients
res = {"ckpt": a.ckpt, "clips": a.clips, "H": H, "eps": a.eps, "n_samples": a.n_samples, "rows": []}
print(f"{'clip':24s} {'sigma':>6s}  cos dJ/du  |wm|/|sim|  cos dJ/dtheta  |wm|/|sim|  split-half cos (estimator)")
for sigma in a.sigmas:
    n = 1 if sigma == 0 else a.n_samples
    us = [[sigma * torch.randn(H, 29, device=dev) if sigma > 0 else torch.zeros(H, 29, device=dev) for _ in range(n)] for _ in range(M)]
    Gs = []  # per start: (n, H, 29)
    per = max(1, a.chunk // len(D))
    jobs = [(m, s) for m in range(M) for s in range(n)]
    G_all = {}
    for i in range(0, len(jobs), per):
        batch = jobs[i : i + per]
        G = fd_grad_batch([m for m, _ in batch], [us[m][s] for m, s in batch])
        for (m, s), g in zip(batch, G):
            G_all[(m, s)] = g
    hs, fs = sim_trajs([m for m, _ in jobs], [us[m][s] for m, s in jobs])
    for m in range(M):
        Gm = torch.stack([G_all[(m, s)] for s in range(n)])
        G_mean = Gm.mean(0)
        # dJ/dtheta: average over samples of the per-sample chain rule along that sample's trajectory
        g_sim = torch.stack([theta_grad(hs, fs, jobs.index((m, s)), G_all[(m, s)]) for s in range(n)]).mean(0)
        split = cos(Gm[: n // 2].mean(0), Gm[n // 2 :].mean(0)) if n > 1 else float("nan")
        row = {"clip": a.clips[m], "sigma": sigma, "cos_dJdu": cos(G_wm[m], G_mean), "norm_ratio_dJdu": float(G_wm[m].norm() / G_mean.norm()),
               "cos_dJdtheta": cos(g_wm[m], g_sim), "norm_ratio_dJdtheta": float(g_wm[m].norm() / g_sim.norm()), "split_half_cos": split,
               "cos_dJdu_per_t": [cos(G_wm[m][t], G_mean[t]) for t in range(H)]}  # fmt: skip
        res["rows"].append(row)
        print(f"{a.clips[m]:24s} {sigma:6.2f}  {row['cos_dJdu']:9.3f}  {row['norm_ratio_dJdu']:10.2f}  {row['cos_dJdtheta']:13.3f}  "
              f"{row['norm_ratio_dJdtheta']:10.2f}  {split:10.3f}", flush=True)  # fmt: skip
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
print("wrote", a.out)
